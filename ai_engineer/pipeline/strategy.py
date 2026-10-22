"""Strategy stage: stop count and compounds, undercut / overcut, rivals' stops, race simulation.

* ``strategy_plan()`` — best plan for the rest of the race (race_strategy.py), built at the
  expected tyre model and at both ends of its range, so the HUD can say if the call survives.
* Rival stop / compound predictions (rival_strategy.py) and the proactive calls they drive:
  react to a nearby rival's stop, pre-empt an undercut, "Hamilton likely to box lap 22".
* Monte Carlo race simulation (race_sim.py) re-run once a lap or on an event, and the Plan B
  call when it beats Plan A (strategy_replan.py).
"""

from __future__ import annotations

import time
from typing import Any

import fitted_models
import pit_wall
import race_sim
import race_strategy
import rival_strategy
import session_modes
import strategy_replan
import track_wetness
from packets import CarStatusData, LapData, SessionData
from pipeline.models import ModelsStage
from pipeline.state import JSON
from voice_engine import Priority


class StrategyStage(ModelsStage):
    # ── Race plan ──

    def _strategy_calibration(self, case: str = 'expected') -> tuple[float, float]:
        """(deg_scale, life_scale) for the expected case, or the ends of the model ranges:
        'worse' = tyres fall off faster and wear out sooner, 'better' = the opposite."""
        status = self.latest_status
        assert status is not None
        prediction = self.pit_prediction()
        deg = self.pace_trend()
        to_limit = prediction.laps_to_latest if prediction else None
        if case != 'expected':
            spread = self.deg_forecast()
            if deg is not None and spread is not None:
                deg = spread.high if case == 'worse' else spread.low
            if prediction is not None and prediction.latest_lap_range:
                assert self.latest_lap is not None
                lap = self.latest_lap.current_lap_num
                soonest, _mid, latest = prediction.latest_lap_range
                bound = soonest if case == 'worse' else latest
                if bound is not None:
                    to_limit = bound - lap
                elif case == 'better':
                    to_limit = None if to_limit is None else to_limit * 1.5
        return race_strategy.calibration(status.compound_name, status.tyres_age_laps, deg, to_limit)

    def strategy_plan(self) -> race_strategy.StrategyPlan | None:
        """Best stop count and compound sequence for the rest of the race, or None outside a race."""
        status, lap, session = self.latest_status, self.latest_lap, self.session
        if status is None or lap is None or session is None or self.session_mode != session_modes.RACE:
            return None
        laps_remaining = session.total_laps - lap.current_lap_num + 1
        key = (
            lap.current_lap_num,
            status.compound_name,
            status.tyres_age_laps,
            frozenset(self.compounds_used),
            session.track_id,
            laps_remaining,
            len(self.session_lap_stats),
        )
        if self._strategy_cache and self._strategy_cache[0] == key:
            cached: race_strategy.StrategyPlan = self._strategy_cache[1]
            return cached
        args = (
            laps_remaining,
            status.compound_name,
            status.tyres_age_laps,
            lap.current_lap_num,
            race_strategy.pit_loss_s(session.track_id),
            self.compounds_used,
        )
        scales = self._strategy_calibration()
        plan = race_strategy.plan_race(*args, *scales)
        # The same plan at both ends of the tyre-model ranges: does the call survive?
        cases = {}
        for case in ('worse', 'better'):
            case_scales = self._strategy_calibration(case)
            cases[case] = plan if case_scales == scales else race_strategy.plan_race(*args, *case_scales)
        self._strategy_cache = (key, plan, scales, cases)
        return plan

    def strategy_uncertainty(self) -> JSON | None:
        """How the plan changes if the tyres turn out at either end of the model's range."""
        plan = self.strategy_plan()
        if plan is None or not self._strategy_cache or len(self._strategy_cache) < 4:
            return None
        cases = self._strategy_cache[3]
        out: JSON = {'level': fitted_models.DEFAULT_LEVEL}
        stops, next_stop = [], []
        for case in ('worse', 'expected', 'better'):
            option = cases.get(case) if case != 'expected' else plan
            option = option.best if option is not None else None
            out[case] = (
                None
                if option is None
                else {'stops': option.stops, 'compounds': option.compounds, 'pit_laps': option.pit_laps}
            )
            if option is not None:
                stops.append(option.stops)
                if option.pit_laps:
                    next_stop.append(option.pit_laps[0])
        out['stops_agree'] = len(set(stops)) <= 1
        out['next_stop_lap_range'] = [min(next_stop), max(next_stop)] if next_stop else None
        return out

    def strategy_scales(self) -> tuple[float, float]:
        """(deg_scale, life_scale) the current plan was built with."""
        if not self._strategy_cache:
            return (1.0, 1.0)
        scales: tuple[float, float] = self._strategy_cache[2]
        return scales

    def _rival_status(self, index: int | None) -> CarStatusData | None:
        if self.all_status is None or index is None or not 0 <= index < len(self.all_status):
            return None
        return self.all_status[index]

    # ── Rival stop / compound predictions ──

    def _race_context(self) -> rival_strategy.RaceContext | None:
        session, lap = self.session, self.latest_lap
        if session is None or self.session_mode != session_modes.RACE or session.total_laps <= 0:
            return None
        window = self.radio.pit_window
        return rival_strategy.RaceContext(
            track_id=session.track_id,
            total_laps=session.total_laps,
            track_length=session.track_length,
            pit_loss_s=race_strategy.pit_loss_s(session.track_id),
            game_window=tuple(window) if window else None,
            player_start_compound=self.player_start_compound,
            player_stops=getattr(lap, 'num_pit_stops', 0) if lap else 0,
            weather=session.weather,
        )

    def rival_prediction(self, index: int | None) -> rival_strategy.RivalPrediction | None:
        """Next stop lap and compound for one rival (None outside a race or without data)."""
        ctx = self._race_context()
        if ctx is None or index is None or index == self.rivals.player_index:
            return None
        return self.rivals.predict(index, ctx, self.fitted, self.radio.driver_name(index))

    def rival_predictions(self) -> list[rival_strategy.RivalPrediction]:
        """Every rival's predicted next stop and compound, in race order."""
        ctx = self._race_context()
        if ctx is None:
            return []
        return self.rivals.predict_all(
            ctx, self.fitted, self.radio.driver_names, exclude=self.rivals.player_index
        )

    def _rival_stop_in_laps(
        self, prediction: rival_strategy.RivalPrediction | None, lap_num: int
    ) -> int | None:
        """Laps a rival is expected to stay out after we box this lap (soonest end of the range)."""
        if prediction is None:
            return None
        if not prediction.will_stop:
            return race_strategy.MAX_COVER_LAPS
        soonest = prediction.stop_lap_range[0]
        return None if soonest is None else soonest - lap_num

    def update_rival_forecasts(
        self, lap: LapData, plan: race_strategy.StrategyPlan, deg_scale: float, life_scale: float
    ) -> None:
        """Proactive calls from the rival stop model, for the cars directly ahead and behind.

        * Car behind due to box this lap or next with an undercut that would work: pre-empt it
          (box now) when our own stop is close, otherwise warn to be ready to cover.
        * Otherwise, when the car ahead or behind is due within a couple of laps, a heads-up:
          "Hamilton likely to box lap 22, onto hards."
        Each car gets at most one call per stint.
        """
        cfg = self.config.strategy
        status = self.latest_status
        assert status is not None
        if lap.current_lap_num < 2 or (self.session and self.session.safety_car_status):
            return
        wet = race_strategy.normalise_compound(status.compound_name) in race_strategy.WET_COMPOUNDS
        preempt_token = ('pre_empt', lap.num_pit_stops)
        for side, index, gap in (
            ('behind', self.radio.car_behind_index, self.radio.gap_behind_s),
            ('ahead', self.radio.car_ahead_index, self.radio.gap_ahead_s),
        ):
            prediction = self.rival_prediction(index)
            if prediction is None or not prediction.will_stop or gap is None or gap > cfg.rival_react_gap_s:
                continue
            due_in = prediction.laps_to_stop(lap.current_lap_num)
            token = (index, prediction.stops_done)
            if token in self.rival_calls or due_in is None:
                continue
            who = f'{prediction.name} {side}' if prediction.name else f'Car {side}'
            onto = f' for {prediction.next_compound.lower()}s' if prediction.next_compound else ''
            if (
                side == 'behind'
                and not wet
                and prediction.basis == 'wear'
                and prediction.prob_stop_by(lap.current_lap_num + cfg.rival_preempt_laps)
                >= cfg.rival_preempt_prob
                and preempt_token not in self.rival_calls
                and plan.best.stops > 0
                and plan.laps_remaining > 3
            ):
                threat = race_strategy.undercut_gain(
                    status.compound_name,
                    status.tyres_age_laps,
                    prediction.next_compound,
                    1,
                    deg_scale,
                    life_scale,
                )
                if threat >= gap - race_strategy.CALL_MARGIN_S:
                    self.rival_calls.update((token, preempt_token))
                    my_stop_in = plan.best.pit_laps[0] - lap.current_lap_num
                    if my_stop_in <= race_strategy.UNDERCUT_WINDOW_LAPS:
                        call = race_strategy.StrategyCall(
                            'pre_empt',
                            f'Expect {who} to box this lap or next{onto}. Undercut threat, '
                            f'box this lap to stay ahead, box box.',
                            threat,
                            gap,
                            index,
                        )
                        self._make_strategy_call(call, 'rival_preempt')
                    else:
                        call = race_strategy.StrategyCall(
                            'cover_ready',
                            f'{who} likely to box lap {prediction.stop_lap}{onto}. '
                            'Undercut would work, be ready to cover.',
                            threat,
                            gap,
                            index,
                        )
                        self._make_strategy_call(call, 'rival_forecast')
                    continue
            if due_in <= cfg.rival_forecast_laps:
                self.rival_calls.add(token)
                call = race_strategy.StrategyCall(
                    'rival_forecast',
                    f'{who} likely to box lap {prediction.stop_lap}{onto}.',
                    None,
                    gap,
                    index,
                )
                self.last_strategy_call = {**call.as_dict(), 'lap': lap.current_lap_num}
                self.speak(call.message, Priority.RACE, key='rival_forecast', ttl=8, cooldown=20)

    def rival_strategy_summary(self) -> JSON | None:
        """Nearby rivals' predicted stops for the HUD and the AI prompt."""
        lap = self.latest_lap
        predictions = self.rival_predictions()
        if lap is None or not predictions:
            return None
        reach = self.config.strategy.rival_summary_positions
        nearby = [p for p in predictions if p.position and abs(p.position - lap.car_position) <= reach]
        out: JSON = {'field': self.rivals.field.as_dict(), 'nearby': [], 'ahead': None, 'behind': None}
        for p in nearby:
            row = {
                k: getattr(p, k)
                for k in (
                    'car_index',
                    'name',
                    'position',
                    'compound',
                    'tyre_age',
                    'wear_pct',
                    'will_stop',
                    'stop_lap',
                    'stop_lap_range',
                    'next_compound',
                    'next_compound_probs',
                    'basis',
                    'summary',
                )
            }
            out['nearby'].append(row)
            if p.car_index == self.radio.car_ahead_index:
                out['ahead'] = row
            elif p.car_index == self.radio.car_behind_index:
                out['behind'] = row
        return out

    def update_strategy_calls(self, lap: LapData) -> None:
        """React to nearby rivals pitting and call an undercut when one is on."""
        events = self.radio.pop_rival_pit_events()
        status = self.latest_status
        if self.session_mode != session_modes.RACE or status is None or lap.pit_status:
            return
        plan = self.strategy_plan()
        if plan is None:
            return
        deg_scale, life_scale = self.strategy_scales()
        for event in events:
            if abs(event['gap_s']) > self.config.strategy.rival_react_gap_s:
                continue
            rival = self._rival_status(event['car_index'])
            # The stop model's last prediction for this car (made before it turned in).
            last = self.rivals.last_prediction(event['car_index'])
            expected = last.next_compound if last is not None else None
            call = race_strategy.react_to_rival_stop(
                'ahead' if event['gap_s'] > 0 else 'behind',
                event['gap_s'],
                status.compound_name,
                status.tyres_age_laps,
                rival.compound_name if rival else None,
                None,
                plan.laps_remaining,
                plan.best.stops,
                deg_scale,
                life_scale,
                event['car_index'],
                rival_name=self.radio.driver_name(event['car_index']),
                assumed_new_compound=expected,
            )
            self._make_strategy_call(call, 'rival_pit')

        self.update_rival_forecasts(lap, plan, deg_scale, life_scale)
        if (
            plan.best.stops == 0
            or race_strategy.normalise_compound(status.compound_name) in race_strategy.WET_COMPOUNDS
        ):
            return  # no undercut maths on a wet track: the crossover decides stops there
        token = (lap.num_pit_stops, lap.current_lap_num)
        if token in self.undercut_calls:
            return
        rival = self._rival_status(self.radio.car_ahead_index)
        if rival is None:
            return
        undercut = race_strategy.undercut_opportunity(
            self.radio.gap_ahead_s,
            status.compound_name,
            status.tyres_age_laps,
            rival.compound_name,
            rival.tyres_age_laps,
            plan.best.pit_laps[0] - lap.current_lap_num,
            plan.laps_remaining,
            deg_scale,
            life_scale,
            self.radio.car_ahead_index,
            rival_stop_in_laps=self._rival_stop_in_laps(
                self.rival_prediction(self.radio.car_ahead_index), lap.current_lap_num
            ),
        )
        if undercut is not None:
            self.undercut_calls.add(token)
            self._make_strategy_call(undercut, 'undercut')

    def _make_strategy_call(self, call: race_strategy.StrategyCall, key: str) -> None:
        lap = self.latest_lap.current_lap_num if self.latest_lap else None
        self.last_strategy_call = {**call.as_dict(), 'lap': lap}
        self.speak(call.message, Priority.STRATEGY, key=key, ttl=10, cooldown=0)

    # ── Monte Carlo race simulation (race_sim.py) ──

    def race_sim_state(self) -> race_sim.RaceState | None:
        """The race as it stands, for the simulator: every running car's gap, tyres and pace."""
        if self.session_mode != session_modes.RACE or self.latest_lap is None:
            return None
        laps = self.field_laps
        if not laps:  # player only (no field data yet)
            laps, index = [self.latest_lap], 0
        else:
            index = self.player_index
        statuses: list[Any] | None = self.all_status if self.all_status else None
        if statuses is None and self.latest_status is not None:
            statuses = [None] * len(laps)
            statuses[index] = self.latest_status
        predictions = None
        rival_predictions = getattr(self, 'rival_predictions', None)
        if callable(rival_predictions):  # rival stop model, when there is one
            try:
                predictions = {p.car_index: p for p in (rival_predictions() or [])}
            except Exception:  # a broken rival model must not stop the sim
                predictions = None
        names = {i: self.radio.driver_name(i) for i in range(len(laps))}
        w, session = self.wetness, self.session
        assert session is not None
        conditions = (
            None
            if w.wetness is None
            else {
                'wetness': w.wetness,
                'wetness_sd': w.sd,
                'weather': session.weather,
                'track_temp': session.track_temperature,
                'forecast': tuple(getattr(session, 'weather_forecast', ())),
                'lap_s': w.lap_s if w.dry_laps else None,
            }
        )
        return race_sim.build_state(
            laps, statuses, index, session, self.field_tracker, names, predictions, self.fitted, conditions
        )

    def update_race_simulation(self) -> None:
        """Re-run the simulation once a lap, and straight away on a Safety Car, VSC, rain or a
        nearby rival's stop; then weigh Plan B against Plan A (strategy_replan.py)."""
        if not self.sim_runner.enabled or self.session_mode != session_modes.RACE:
            return
        status, lap, session = self.latest_status, self.latest_lap, self.session
        if status is None or lap is None or session is None:
            return
        if lap.result_status == 3 or (session.total_laps > 0 and lap.current_lap_num > session.total_laps):
            self._settle_race_odds(lap, status, session)
            return
        replanner = self.replanner
        families = None
        if self.all_status:
            families = {
                i: track_wetness.family(st.compound_name)
                for i, st in enumerate(self.all_status)
                if st is not None
                and (
                    not self.field_laps or i >= len(self.field_laps) or self.field_laps[i].result_status == 2
                )
            }
        best = track_wetness.best_family(self.wetness.wetness) if self.wetness.wetness is not None else None
        replanner.observe(
            lap.current_lap_num,
            session.safety_car_status,
            session.weather,
            self.field_laps,
            self.player_index,
            self.radio.driver_name,
            best,
            families,
        )
        self._consider_plan_b()
        if lap.pit_status:
            return
        wet = session.weather >= strategy_replan.WET_WEATHER
        key = (
            lap.current_lap_num,
            session.safety_car_status,
            lap.num_pit_stops,
            status.compound_name,
            (wet, replanner.seen_best),
            replanner.events,
        )
        if key == self.sim_runner.latest_key:
            return
        now = time.monotonic()
        last_key = self._last_sim_key
        event_only = last_key is not None and key[:5] == last_key[:5]
        if event_only and now - self._last_event_submit < strategy_replan.MIN_EVENT_INTERVAL_S:
            return  # only an event changed, and one re-ran a moment ago
        plan = self.strategy_plan()
        laps_remaining = session.total_laps - lap.current_lap_num + 1
        if plan is None and laps_remaining <= 0:
            return
        candidates: list[Any]
        if plan is not None:
            replanner.sync_plan_a(
                plan, lap.current_lap_num, lap.num_pit_stops, status.compound_name, plan.laps_remaining
            )
            last_lap = session.total_laps - 1
            trajectory = (
                self.wetness.trajectory(max(plan.laps_remaining, 1))
                if self.wetness.wetness is not None
                else None
            )
            if trajectory is not None and max(trajectory) < 0.02 and not wet:
                trajectory = None  # dry and staying dry: no tyre-type options needed
            candidates = replanner.candidates(
                plan,
                lap.current_lap_num,
                session.safety_car_status in (1, 2),
                wet,
                status.compound_name,
                trajectory,
                last_lap,
            )
        else:
            candidates = []
        if not candidates:
            # Nothing left to plan (the final lap): still run the race to the flag on this set, so
            # the odds follow track position instead of freezing on an older lap's run.
            candidates = [race_sim.stay_out(status.compound_name)]
            plan = None
        state = self.race_sim_state()
        if state is None:
            return
        deg_scale, life_scale = self.strategy_scales()
        seed = (
            lap.current_lap_num * 1009
            + session.safety_car_status * 101
            + lap.num_pit_stops
            + replanner.events * 7919
        ) & 0xFFFFFFFF
        reference = 0 if replanner.plan_a is not None and plan is not None else None
        if self.sim_runner.submit(
            key,
            state,
            candidates,
            reference=reference,
            models=self.fitted,
            seed=seed,
            pit_loss_mean=race_strategy.pit_loss_s(session.track_id),
            player_deg_scale=deg_scale,
            player_life_scale=life_scale,
            player_deg=self.deg_forecast(),
        ):
            self._last_sim_key = key
            if event_only:
                self._last_event_submit = now
            if reference is not None:
                replanner.note_submitted(key)
            self._consider_plan_b()  # 'sync' mode: the result is already in

    def _settle_race_odds(self, lap: LapData, status: CarStatusData, session: SessionData) -> None:
        """After the chequered flag the result is known: the odds become certain."""
        position = lap.car_position
        if position <= 0:
            return
        laps = self.field_laps or []
        cars = sum(1 for car in laps if car.car_position > 0) or position
        key = ('finished', position, cars)
        if self.sim_runner.latest_key == key:
            return
        self.sim_runner.settle(
            key,
            race_sim.settled_result(
                position, cars, min(lap.current_lap_num, session.total_laps), status.compound_name
            ),
        )

    def _consider_plan_b(self) -> None:
        """Call Plan B when the latest re-run says it beats Plan A by the set margin."""
        lap = self.latest_lap
        if lap is None:
            return
        runner = self.sim_runner
        call = self.replanner.consider(runner.latest, runner.result_key, lap.current_lap_num)
        if call is None:
            return
        self.last_strategy_call = {
            **call.as_dict(),
            'gain_s': call.time_gain_s,
            'gap_s': None,
            'car_index': None,
        }
        self.speak(call.message, Priority.STRATEGY, key='plan_b', ttl=12, cooldown=0)

    def race_simulation(self) -> JSON | None:
        """Latest simulation summary, or None when it is off, not run yet or out of date."""
        result = self.sim_runner.latest
        lap = self.latest_lap
        if result is None or lap is None or self.session_mode != session_modes.RACE:
            return None
        if lap.current_lap_num - result.lap > 3:
            return None
        summary: JSON = result.summary()
        return summary

    def strategy_odds(self) -> JSON | None:
        """The latest race simulation as win/podium/points odds and plan odds for the HUD,
        with how the odds moved over the race (one point per simulated lap)."""
        if self.race_simulation() is None:
            return None
        result = self.sim_runner.latest
        assert result is not None
        if result is not self._odds_result:
            self._odds_result = result
            top = result.outcomes[0] if result.outcomes else None
            if top is not None:
                point = {
                    'lap': result.lap,
                    'p_win': top.p_win,
                    'p_podium': top.p_podium,
                    'p_points': top.p_points,
                }
                self.odds_history = [p for p in self.odds_history if p['lap'] < result.lap] + [point]
            self._odds = pit_wall.strategy_odds(result, self.odds_history)
        return self._odds

    def strategy_summary(self) -> JSON | None:
        """Plan plus the latest undercut/overcut call, for the HUD and the AI prompt."""
        plan = self.strategy_plan()
        if plan is None:
            return None
        summary: JSON = plan.as_dict()
        summary['radio'] = race_strategy.describe_plan(plan)
        uncertainty = self.strategy_uncertainty()
        summary['uncertainty'] = uncertainty
        if uncertainty and not uncertainty['stops_agree'] and uncertainty['worse']:
            worse = uncertainty['worse']['stops']
            if worse != plan.best.stops:
                words = {0: 'no stop', 1: 'one stop', 2: 'two stops', 3: 'three stops'}
                summary['radio'] += f" If the tyres go off faster, it's {words.get(worse, worse)}."
        simulation = self.race_simulation()
        summary['simulation'] = simulation
        if simulation and simulation.get('radio'):
            summary['radio'] += ' ' + simulation['radio']
        lap = self.latest_lap
        replan = (
            self.replanner.summary(lap.current_lap_num if lap else None) if self.sim_runner.enabled else None
        )
        summary['replan'] = replan
        summary['conditions'] = self.conditions_summary()
        if replan:
            a = replan['plan_a']
            if (a['compounds'], a['pit_laps']) != (plan.best.compounds, plan.best.pit_laps):
                summary['radio'] += ' ' + replan['radio']
        summary['last_call'] = self.last_strategy_call
        status, rival = self.latest_status, self._rival_status(self.radio.car_ahead_index)
        wet = (
            race_strategy.normalise_compound(status.compound_name if status else None)
            in race_strategy.WET_COMPOUNDS
        )
        if self.radio.gap_ahead_s is not None and rival is not None and status is not None and not wet:
            deg_scale, life_scale = self.strategy_scales()
            assert lap is not None
            cover = race_strategy.cover_laps(
                self._rival_stop_in_laps(
                    self.rival_prediction(self.radio.car_ahead_index), lap.current_lap_num
                ),
                plan.laps_remaining,
            )
            gain = race_strategy.undercut_gain(
                rival.compound_name,
                rival.tyres_age_laps,
                race_strategy.next_compound(status.compound_name),
                cover,
                deg_scale,
                life_scale,
            )
            summary['undercut_on_car_ahead'] = {
                'gap_s': round(self.radio.gap_ahead_s, 2),
                'gain_s': gain,
                'works': gain - self.radio.gap_ahead_s >= race_strategy.CALL_MARGIN_S,
                'rival_compound': rival.compound_name,
                'rival_tyre_age': rival.tyres_age_laps,
                'rival_stays_out_laps': cover,
            }
        else:
            summary['undercut_on_car_ahead'] = None
        rivals = self.rival_strategy_summary()
        summary['rivals'] = rivals
        lines = [
            f"{side.capitalize()}: {rivals[side]['summary']}"
            for side in ('ahead', 'behind')
            if rivals and rivals.get(side)
        ]
        summary['rival_radio'] = ' '.join(lines) or None
        if self.session and self.session.pit_stop_rejoin_position:
            summary['game_rejoin_position'] = self.session.pit_stop_rejoin_position
        return summary

    def conditions_summary(self) -> JSON | None:
        """Track wetness, quickest tyre, and when to change tyre type; None on a dry, dry-forecast day."""
        status = self.latest_status
        fam = track_wetness.family(status.compound_name) if status else None
        out: JSON | None = self.wetness.summary(fam)
        if out is None:
            return None
        rain_coming = any(f['rain_pct'] >= 30 or f['weather'] >= 3 for f in out['forecast_rain'])
        if (
            out['wetness'] < 0.02
            and out['trend'] != 'wetting'
            and not rain_coming
            and fam != 'inter'
            and fam != 'wet'
        ):
            return None
        out['radio'] = self.wetness.radio(fam)
        return out
