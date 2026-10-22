"""Models stage: tyre wear, degradation, fuel and the instant tyre-health read.

Two complementary sources, kept separate on purpose:

* fitted_models.py / strategy_ml.py learn from laps (stored fits, earlier sessions in the
  history database, then this stint's own laps) and answer with ranges.
* physics_models.py scores the tyres from the first frame of a fresh set, no laps needed.

This stage only calls into them and caches the results per lap; it says nothing.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import fitted_models
import physics_models
import session_modes
import strategy_ml
from log_setup import get_logger
from pipeline.radio import RadioStage
from pipeline.state import JSON

log = get_logger('models')


class ModelsStage(RadioStage):

    def _past_records(self) -> list[fitted_models.LapRecord]:
        """Laps from earlier sessions on this track that the fitted store has not seen yet."""
        if self._history_records is None:
            records: list[fitted_models.LapRecord] = []
            if self.history and self.session:
                try:
                    records, _ = fitted_models.laps_from_history(
                        self.history, self.fitted.consumed_captures | self._session_captures,
                        track_id=self.session.track_id)
                except sqlite3.Error as error:
                    log.warning('Could not load lap history: %s', error)
                records = [r for r in records if r.track_length == self.session.track_length]
            self._history_records = records
        return self._history_records

    def _session_records(self) -> list[fitted_models.LapRecord]:
        s = self.session
        if s is None:
            return []
        return fitted_models.records_from_lap_rows(
            self.session_lap_stats, source='live', session='live', track_id=s.track_id,
            track_length=s.track_length, session_type=s.session_type,
            race_laps=self._race_laps(), wet=s.weather >= 3)

    def _race_laps(self) -> int:
        s = self.session
        return s.total_laps if s and session_modes.has_pit_strategy(self.session_mode) else 0

    def model_forecast(self) -> JSON | None:
        """Wear, degradation and fuel models for the car's current tyres and track.

        Each model is the stored fit (or its fallback), refined with earlier sessions in the
        history database and then with this stint's / session's own laps. Cached per lap.
        """
        status, session = self.latest_status, self.session
        if status is None or session is None:
            return None
        key = (status.actual_tyre_compound, status.tyres_age_laps, len(self.session_lap_stats),
               session.track_id, session.weather >= 3)
        if self._forecast_cache and self._forecast_cache[0] == key:
            return self._forecast_cache[1]
        compound = status.actual_tyre_compound
        past = list(self._past_records())
        session_records = self._session_records()
        stint = fitted_models.current_stint(session_records, compound, status.tyres_age_laps)
        ckey = fitted_models.compound_key(compound)
        stint_ids = {x.stint for x in stint}
        same_compound_past = [r for r in past + session_records
                              if r.compound == ckey and r.stint not in stint_ids]

        wear_model = fitted_models.refine(
            self.fitted.wear_model(session.track_id, compound, self._race_laps()), same_compound_past)
        stint_wear = wear_model.condition(*fitted_models.stint_increments(stint))
        deg_model = fitted_models.refine(self.fitted.deg_model(session.track_id, compound),
                                         same_compound_past)
        deg_ages, deg_times = fitted_models.stint_times(stint)
        deg = deg_model.condition(deg_ages, deg_times)
        fuel_model = fitted_models.refine(
            self.fitted.fuel_model(session.track_id, session.track_length, self._race_laps()),
            [r for r in past if r.session != 'live'])
        session_fuel = fuel_model.condition([u for r in session_records
                                             if (u := r.fuel_used) is not None])
        forecast: JSON = {'stint_wear': stint_wear, 'deg': deg, 'deg_laps': len(deg_ages),
                          'deg_basis': deg_model.basis, 'deg_fit_laps': deg_model.n_laps,
                          'fuel': session_fuel, 'compound': ckey}
        self._forecast_cache = (key, forecast)
        return forecast

    def wear_forecast(self) -> fitted_models.StintWear | None:
        """The wear model conditioned on this stint, or None until there is any data at all."""
        forecast = self.model_forecast()
        if forecast is None:
            return None
        stint: fitted_models.StintWear = forecast['stint_wear']
        if stint.model.n_laps == 0 and stint.stint_laps < 2:
            return None             # built-in estimate only: not enough to quote a window from
        return stint

    def tyre_model(self) -> fitted_models.StintWear | None:
        """Backwards-compatible name for the current stint's wear forecast."""
        return self.wear_forecast()

    def _worst_wear(self) -> float:
        return max(self.latest_damage.tyres_wear) if self.latest_damage else 0.0

    def pit_prediction(self) -> strategy_ml.PitPrediction | None:
        """Pit window (lap ranges) from the fitted wear model, or None until there is data."""
        status, lap, session = self.latest_status, self.latest_lap, self.session
        if status is None or lap is None or session is None:
            return None
        if not session_modes.has_pit_strategy(self.session_mode):
            return None     # no stops in qualifying or time trial
        key = (lap.current_lap_num, status.actual_tyre_compound, status.tyres_age_laps,
               len(self.session_lap_stats))
        if self._ml_cache and self._ml_cache[0] == key:
            cached: strategy_ml.PitPrediction | None = self._ml_cache[1]
            return cached
        prediction = strategy_ml.predict_pit_window(
            self.wear_forecast(), status.tyres_age_laps, self._worst_wear(), lap.current_lap_num,
            session.total_laps)
        self._ml_cache = (key, prediction)
        return prediction

    def fuel_prediction(self) -> JSON | None:
        """Fuel burn per lap and the margin at the flag, as ranges (low end = the worry)."""
        status = self.latest_status
        if status is None:
            return None
        result: JSON = {
            'fuel_kg': round(status.fuel_in_tank, 1), 'per_lap_kg': None,
            'game_laps_remaining': round(status.fuel_remaining_laps, 1), 'laps_of_fuel': None,
            'margin_laps': None, 'per_lap_kg_range': None, 'laps_of_fuel_range': None,
            'margin_laps_range': None, 'margin_kg_range': None, 'basis': None,
            'level': fitted_models.DEFAULT_LEVEL}
        forecast = self.model_forecast()
        if forecast is None:
            return result
        fuel = forecast['fuel']
        result['basis'] = fuel.model.basis
        result['fit_laps'] = fuel.model.n_laps
        result['session_laps'] = fuel.session_laps
        per = fuel.per_lap()
        laps_of_fuel = fuel.laps_of_fuel(status.fuel_in_tank)
        result['per_lap_kg_range'] = per.as_list(3)
        result['laps_of_fuel_range'] = laps_of_fuel.as_list(1)
        if fuel.model.n_laps or fuel.session_laps:     # a fit or this session's laps: quote it
            result['per_lap_kg'] = round(per.mid, 3)
            result['laps_of_fuel'] = round(laps_of_fuel.mid, 1)
        lap, session = self.latest_lap, self.session
        if lap and session and session.total_laps > 0:
            to_go = session.total_laps - lap.current_lap_num + 1
            kg, laps = fuel.margin(status.fuel_in_tank, to_go)
            result['margin_kg_range'] = kg.as_list(2)
            result['margin_laps_range'] = laps.as_list(1)
            if result['per_lap_kg'] is not None:
                result['margin_laps'] = round(laps.mid, 1)
        return result

    def deg_forecast(self) -> fitted_models.Range | None:
        forecast = self.model_forecast()
        if not forecast:
            return None
        deg: fitted_models.Range = forecast['deg']
        return deg

    def pace_trend(self) -> float | None:
        """Fuel-corrected tyre fall-off in s/lap for the current stint (fitted model + stint laps).

        None until the stint has three clean laps and there is no fitted model to lean on.
        """
        forecast = self.model_forecast()
        if forecast is None:
            return None
        if forecast['deg_laps'] < 3 and forecast['deg_fit_laps'] == 0:
            return None
        return round(float(forecast['deg'].mid), 3)

    def model_summary(self) -> JSON | None:
        """Ranges for the HUD and the AI prompt: wear, degradation, pit window, fuel."""
        forecast = self.model_forecast()
        status = self.latest_status
        if forecast is None or status is None:
            return None
        stint = forecast['stint_wear']
        model = stint.model
        age, wear = status.tyres_age_laps, self._worst_wear()
        prediction = self.pit_prediction()
        fuel: dict[str, Any] = self.fuel_prediction() or {}
        return {
            'level': fitted_models.DEFAULT_LEVEL,
            'compound': forecast['compound'], 'compound_name': status.compound_name,
            'basis': model.basis, 'fit_laps': model.n_laps, 'fit_stints': model.n_stints,
            'stint_laps': stint.stint_laps, 'simulated_fit': bool(model.simulated and model.n_laps),
            'wear_now_pct': round(wear, 1),
            'wear_per_lap_pct': stint.rate(age).as_list(2),
            'wear_in_5_laps_pct': stint.wear_after(age, wear, 5).as_list(1),
            'deg_s_per_lap': forecast['deg'].as_list(3), 'deg_basis': forecast['deg_basis'],
            'pit_ideal_lap': prediction.ideal_lap_range if prediction else None,
            'pit_latest_lap': prediction.latest_lap_range if prediction else None,
            'finish_without_stop_prob': prediction.finish_without_stop_prob if prediction else None,
            'fuel_per_lap_kg': fuel.get('per_lap_kg_range'),
            'fuel_margin_laps': fuel.get('margin_laps_range'),
            'fuel_basis': fuel.get('basis'),
        }

    def tire_health(self) -> physics_models.TireResult | None:
        """Instant physics-model tyre read: grip, health and overheat risk, no lap history needed.

        Complements pit_prediction() (which learns a wear curve from observed laps and needs
        several of them): this scores the current instant from a model of the compound, so it
        is available from the first telemetry frame of a fresh set.
        """
        if self.latest_status is None:
            return None
        return physics_models.evaluate_tire(
            self.latest_status.compound_name,
            self.latest_status.tyres_age_laps,
            self.latest_damage.tyres_wear if self.latest_damage else None,
            self.latest_telemetry.tyres_surface_temperature if self.latest_telemetry else None)
