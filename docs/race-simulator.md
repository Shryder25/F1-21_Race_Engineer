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
