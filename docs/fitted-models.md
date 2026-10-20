# Tyre and fuel models fitted to the game

`ai_engineer/fitted_models.py` replaces the single-number tyre and fuel estimates with models
fitted to F1 2021 itself, **per tyre compound and per track**, from recorded stints. Every
prediction is a range: **[low, expected, high]**, by default the 10th to 90th percentile (an
80% range), so strategy can see how sure the engineer is.

## What is modelled

| Model | Per | Equation | Spread |
|---|---|---|---|
| Tyre wear | track + compound | worst-tyre wear added in one lap = `r0 + r1 * tyre age` | lap-to-lap noise, plus a stint-to-stint effect (each set wears a bit differently) |
| Fall-off | track + compound | fuel-corrected lap time lost per lap of tyre age (s/lap) | stint-to-stint spread of the slope |
| Fuel | track | kg burned per racing lap | lap-to-lap noise, plus a session-to-session effect (fuel mix, lift and coast) |

Compounds are the game's actual compounds (`C1`-`C5`, `INTER`, `WET`), not Soft/Medium/Hard,
because Soft is a different compound at different tracks. The fitted parameters carry their
own uncertainty (a posterior covariance), so a fit from two stints gives wider ranges than a
fit from twenty.

## Fallbacks when data is thin

Each lookup takes the most specific model that has data:

1. **This track** - fitted from your stints on this track and compound.
2. **Other tracks (pooled)** - the compound fitted across every track, after dividing out each
   track's tyre severity (track database: Low 0.8, Medium 1.0, High 1.15, Very High 1.3), then
   scaled back up to this track with an extra 25% between-track spread. Fuel pools in kg per km.
3. **Built-in estimate** - a wide prior (about +/-50%) from typical F1 2021 wear rates, the
   physics model's ageing rates and 0.33 kg/km of fuel.

Then, live, the engineer narrows that model with what it is seeing right now:

* laps from earlier sessions in the history database that the fitted file has not seen yet;
* this stint's own laps (tyre wear and fall-off) and this race's own laps (fuel), as a
  Bayesian update. After three or four laps the stint's own behaviour dominates.

The pit window is only quoted once there is real data (a fit, history, or two laps of this
stint); the built-in estimate alone is too vague to name a lap.

A fit taken at one race distance used at another is widened by 25%, because the game may scale
wear and fuel with race length. If you race different distances, record each.

## Where the ranges go

* **Pit window**: soonest / expected / latest lap for the 60% (target) and 75% (latest) wear
  marks, and the chance of reaching the flag without stopping.
* **Fuel**: burn per lap, laps of fuel and margin at the flag, each as a range (the low end is
  the worrying case).
* **Race strategy**: the plan is also run with the tyres at the bad end of the range (faster
  fall-off, shorter life) and the good end. The summary says whether the stop count holds, and
  the radio line adds "If the tyres go off faster, it's two stops" when it does not.
* **AI prompt**, **voice answers** ("Box around lap 19, window lap 17 to 21") and the
  **HUD's Tyre & Fuel Model card**.

## Fitting

```bash
python main.py --record recordings/spain_race.f1rec.gz       # race, practice long runs, ...
python tools/fit_models.py recordings/*.f1rec.gz --history telemetry-history/telemetry.sqlite3
python main.py                                               # loads telemetry-history/fitted_models.json
```

`--models FILE` points `main.py` at another file. `--report FILE` saves the fit report:
every model with its range, plus a **hold-out check** that fits on every session but one and
replays the one left out lap by lap, counting how often the truth lands inside the range
(close to 80% means the ranges are honest).

From recordings, tyre age, worst-tyre wear and fuel are read 400 m into each lap (past the pit
exit, away from the line where packets disagree about the lap). Pit laps and Safety Car laps
are not counted as racing laps for fuel or pace, lap 1 of a race is not a pace lap, and slick
laps on a wet track stay out of the slick fits. Flashbacks rewind the laps they undo.

`python tools/fit_models.py --simulated` runs the whole pipeline on noisy simulator races with
known true values (`recordings/simulated/fit/`). On those, every true wear rate and fuel burn
lands inside its fitted range and the hold-out ranges catch 79-83% of the truth. That shows the
maths works; **the real fits need your own F1 2021 recordings**.

## File format

`fitted_models.json` (version 1):

```json
{
  "version": 1, "created": "...", "simulated_only": false,
  "sources": [{"path": "recordings/spain.f1rec.gz", "laps": 52, "simulated": false}],
  "consumed_captures": ["<history capture ids already in the fit>"],
  "wear": {"4/C3": {"compound": "C3", "track_id": 4, "mean": [r0, r1], "cov": [[...], [...]],
                    "sigma2": 0.04, "tau2": 0.09, "n_laps": 48, "n_stints": 4, "basis": "track",
                    "simulated": false, "race_laps": [30]},
           "*/C3": {"...": "pooled, severity-normalised"}},
  "deg":  {"4/C3": {"mean": 0.06, "var": 0.0004, "sigma2": 0.1, "tau2": 0.0009, "...": "..."}},
  "fuel": {"4": {"mean": 1.75, "var": 0.001, "sigma2": 0.004, "tau2": 0.006, "...": "..."},
           "*": {"...": "pooled, kg per km"}}
}
```
