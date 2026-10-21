# Monte Carlo race simulator

`ai_engineer/race_sim.py` turns the strategy model's candidate plans into odds by playing
the rest of the race many times for the whole field.

## What one run does

For each lap from the lap in progress to the flag, every running car gets a lap time:

```
base pace + driver form + compound offset + fall-off x tyre age
          + cliff past tyre life + fuel load x 0.03 s/kg + lap noise
          + dirty air / held up + pit loss on the lap it boxes
```

Then the field is walked front to back. A car that would cross the line within 0.3 s of the
nearest on-track car ahead has caught it and passes with a probability that rises with its
pace advantage, centred on the track's threshold (below); otherwise it is held 0.3 s behind.
A car that gets by may take one more car the same lap. Cars on their in-lap neither block nor
get blocked.

| Ingredient | Source | Notes |
|---|---|---|
| Tyre fall-off (s/lap) and spread | `fitted_models.ModelStore.deg_model` per actual compound and track | each stint draws its own slope; the player's current stint uses its stint-conditioned range |
| Tyre life | physics life per compound, scaled by a draw from the fitted wear model's spread | the player's `life_scale` calibration applies to the player only |
| Fuel burn | `fitted_models` fuel model | drawn per car per run |
| Safety Car chance | `track_profiles` `sc_probability` (share of races with an SC) | Poisson rate per lap, lap 1 four times likelier; lasts 3-5 laps; field bunches to 0.8 s gaps |
| VSC chance | `vsc_probability` if a calibration provides it, else 0.7 x SC chance | **estimate**, lasts 1-3 laps, gaps frozen |
| Stops under SC/VSC | any car with a planned stop within 8 laps takes it when a neutralisation starts | costs 55% (SC) / 65% (VSC) of a green stop |
| Pit loss | `race_strategy.pit_loss_s` per track as the mean | + normal spread (sd 0.6 s) + 6% slow stops (+3 s on average, exponential) |
| Overtaking | `track_profiles` `overtaking` rating 1-5 | coin-flip pace advantage 0.3 / 0.6 / 0.9 / 1.3 / 2.2 s/lap |
| Rain | `track_profiles` `rain_probability` x share of race left | arrives once and stays; slicks lose 7 s/lap, everyone boxes for inters within a lap |
| Rival plans | `race_strategy.plan_race` per rival (25% run their second-best stop count), stop laps jittered sd 1.5 laps | or a rival-stop prediction object (`stop_lap_mean`, `stop_lap_sd`, `will_stop`, `next_compound_probs`) when supplied |
| Rival pace | median of each car's clean laps (no lap 1, in-laps, out-laps or SC laps), corrected for tyre age and fuel | cars with no clean laps get a guess from track position and a wider form spread |

Every random number is drawn once per run and shared by all candidate plans (common random
numbers), so plans are compared under the same Safety Cars, slow stops and rival form.

## Outputs

Per plan: share of runs where it gave the best result of all plans (`p_best`, the
"two-stop wins 60% of the time" number), P(win), P(podium), P(points), the finishing-position
distribution and its 10th/50th/90th percentile, and race time against the best plan in the
same run. Per car, under the recommended plan: mean position, P(win), P(podium). Plus the
share of runs with an SC, VSC or rain, and the pit-loss spread.

Live, `AnalysisEngine` re-runs it on a background thread once a lap and whenever the Safety
Car status, the player's stop count or compound changes (`race_simulation='async'`, which
`main.py` sets; tests use `'sync'`, everything else defaults to `'off'` so golden replays are
unchanged). It shows in the strategy summary (`strategy_summary()['simulation']`), the HUD
strategy card, the AI prompt and the strategy voice answer. A result more than three laps old
is not quoted.

## Live re-planning: Plan A and Plan B (`strategy_replan.py`)

The simulator re-runs once a lap, and also straight away when something changes the race:
a Safety Car or VSC, rain arriving, or a rival within three places boxing (rival-stop
re-runs are spaced at least 3 s apart so a pit cascade doesn't queue one per car).

* **Plan A** is the plan we are on: the first race plan, moved along as we stop (a stop we
  skip slides to the current lap, an unplanned stop or a flashback starts it again). It only
  changes when the engineer calls Plan B.
* Every re-run plays Plan A first, then the strategy model's options, then event options:
  Plan A's next stop brought forward to this lap under SC/VSC or after a nearby rival stops,
  and "box now for inters" when it is raining and we are on slicks. At most six plans per run.
* The simulator compares each plan with Plan A in the same races (`vs_reference`): median
  seconds gained, mean places gained, and the share of runs where it finishes ahead.
* **Plan B is called** only when it gains at least `REPLAN_MARGIN_S` (3 s, `--plan-b-margin`)
  **or** `REPLAN_MARGIN_POSITIONS` (0.5 places, `--plan-b-places`), **and** is ahead in at least
  60% of runs. On a routine lap re-run it must also win two re-runs in a row and come at least
  3 laps after the last call; after an SC, VSC, rain or rival-stop trigger one run is enough.
  The same tyres with stops within two laps count as the same plan (no call), unless one of them
  says box this lap.
* The call is a short `plan_b` radio line, which also plays in quiet mode: *"Safety car. Plan B,
  Plan B. Box this lap for hards. Box, box. Gains about 8 seconds."* The HUD strategy card shows
  it on top for two laps, plus Plan A and any alternative the engine is watching.

### Wet and drying tracks (`track_wetness.py`)

F1 2021 doesn't send how wet the track is, so the engineer estimates it the way a pit wall does:

* **Physics.** Rain soaks the track towards a level set by the rain (light 0.5, heavy 0.8,
  storm 0.95 on a 0-1 scale, about 3 minutes to get most of the way). Once it stops, the track
  dries at about 0.03 per minute at 30 C track temperature, faster when it's hotter (+3%/C) or
  sunny, slower under cloud. A track can only get wetter while it rains.
* **The game's forecast** (session packet, rain chance per sample, now parsed into
  `SessionData.weather_forecast`) sets the rain in each simulated race: one draw per run, so a
  run that gets the shower keeps it for as long as the forecast does. With no forecast, a shower
  falling now lasts about 20 minutes on average, and new rain uses the track's rain chance.
* **Lap times as the sensor.** When cars on slicks and on inters (or inters and wets) lap at the
  same time, the gap between them pins the wetness down. With one tyre type out, each car's
  lap is compared with its own earlier dry laps (fuel-corrected). Each reading is a Bayesian
  update on a grid, and without rain it can only make the track drier than the physics says
  (a dry line), never wetter.

Lap-time cost per tyre type against wetness (a 90 s lap, scaled to the track):

| wetness | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.85 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| slicks | 0 | 1.8 | 4.5 | 9 | 18 | 28 | 34 | 44 |
| inters | 4 | 4 | 4.5 | 5.5 | 8.5 | 13 | 18 | 24 |
| wets | 8 | 8 | 8.5 | 9 | 11 | 13 | 16 | 19 |

So slicks and inters cross at 20% wet, and inters and wets at 70%. Inters below 15% wet and wets
below 35% also overheat and wear three times faster. In the simulator each car (rivals, and us
after a shower the plan couldn't foresee) changes tyre type once the other type is quicker by its
own threshold (about 1.2 s/lap, drawn per car and run, so some gamble early), if the laps left
pay for the stop. Our own changes otherwise follow the plan, and the re-planner offers plans that
switch a lap before, at, and two laps after the expected crossover. Passing the crossover, or any
car changing tyre type ("first car on slicks"), triggers an immediate re-run.

**What would make this match the game exactly:** the table and the drying rate are estimates.
Recordings of real F1 2021 wet races would let both be fitted, the same way `tools/fit_models.py`
fits tyre wear.

Cost: about 0.1-0.2 s per re-run on the 20-car golden replays, about 0.5 s for a 20-car
field with six plans in the unit test, all on the simulator's own thread. The HUD only gets a
small summary dict.

## Validation so far, and what real recordings would sharpen

The sandbox has no F1 2021 game data, so `tests/test_race_sim.py` checks the mechanics on
made-up fields: probabilities add up, the SC rate over a full race matches the profile, VSC
and rain rates follow their inputs, a cheap stop under SC pays, the same plan twice ties
exactly, a clearly worse plan loses, passing is much harder at an overtaking-5 track than an
overtaking-1 track, a held-up car crosses the line just behind, a car on its in-lap does not
block, fitted models and rival predictions flow through.

Constants that are educated guesses and should be fitted from your recordings
(`python main.py --record ...`), roughly in order of how much they move the odds:

1. **Overtaking thresholds** (`PASS_THRESHOLD_S`): from recorded races, count passes against
   the pace gap between the two cars over the previous laps, per track.
2. **SC/VSC rates in the game**: F1 2021 AI incidents are not real-world incidents; count
   neutralisations per race in recordings (the Session packet carries the SC status) and
   replace `sc_probability` / `VSC_SHARE_OF_SC`.
3. **Pit-loss spread**: measure entry-to-exit loss per stop from Lap Data (`pit_lane_time_ms`)
   to replace `PIT_LOSS_SD_S`, `SLOW_STOP_PROB` and the per-track means.
4. **Dirty air and driver form** (`DIRTY_AIR_S`, `DRIVER_FORM_SD_S`): lap-time loss when within
   1 s of a car, and lap-to-lap pace spread of AI cars.
5. **Tyre and fuel inputs**: already fitted from recordings by `tools/fit_models.py`; more
   recordings narrow them and the simulation's spread with them.

Known simplifications: no retirements or damage, SC bunching is instant, rain arrives once
and stays, lapped cars never interact with the leaders (blue flags), only the player's
candidate plans come from the strategy model's best plan per stop count.
