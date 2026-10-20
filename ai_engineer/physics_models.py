"""Physics-based tire, fuel and opponent-threat models.

Adapted from the tire-degradation and threat-scoring approach in the Box-Box
race engineer project (github.com/ploverbyHB/Box-Box, engine_models.py),
rewritten against this repo's F1 2021 packet dataclasses instead of loose
JSON dicts. Unlike strategy_ml.py (which *learns* a wear curve from lap
history), this module scores the *current instant* from a physical model of
each compound, so it works from the very first lap with no history needed —
the two are complementary, not competing: strategy_ml predicts when to pit
from observed data, this module explains how healthy the tyre is right now
and who on track is a threat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class OverheatRisk(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class CompoundParams:
    alpha: float                # linear ageing degradation, s/lap
    beta: float                 # quadratic ageing degradation, s/lap^2
    kappa: float                # lap-time cost of full wear, s
    a_wear: float                # grip sensitivity to wear
    t_star: float                # ideal tyre temperature, C
    sigma: float                 # thermal window half-width, C
    t_hot: float                  # overheat penalty onset, C
    t_crit: float                 # critical overheating temperature, C
    t_cold: float                 # cold/graining penalty onset, C
    expected_life_laps: float    # expected competitive life, laps
    wear_cliff: float            # wear fraction (0-1) at which the tyre falls off a cliff
    competitive_loss_s: float    # cumulative loss (s) at which the tyre is no longer competitive


# Coefficients per F1 2021/24 compound (same source values used by Box-Box: formula1.wiki,
# simracingsetup community data). C0-C5 are the underlying slick compounds; SOFT/MEDIUM/HARD
# are the three a weekend actually allocates, mapped onto the nearest C-compound.
_COMPOUND_PARAMS: dict[str, CompoundParams] = {
    "C0": CompoundParams(0.012, 0.00014, 1.40, 0.20, 100, 8.75, 110, 120, 85, 33, 0.95, 1.40),
    "C1": CompoundParams(0.015, 0.00018, 1.45, 0.22, 100, 8.75, 110, 120, 85, 30, 0.94, 1.45),
    "C2": CompoundParams(0.020, 0.00025, 1.50, 0.25, 95, 7.50, 105, 115, 85, 25, 0.93, 1.50),
    "C3": CompoundParams(0.025, 0.00035, 1.60, 0.28, 90, 7.50, 100, 110, 80, 20, 0.92, 1.60),
    "C4": CompoundParams(0.030, 0.00050, 1.75, 0.32, 90, 6.25, 97, 105, 80, 16, 0.90, 1.75),
    "C5": CompoundParams(0.035, 0.00070, 1.90, 0.36, 85, 7.50, 95, 105, 75, 12, 0.88, 1.90),
    "SOFT": CompoundParams(0.030, 0.00050, 1.75, 0.32, 90, 6.25, 97, 105, 80, 16, 0.90, 1.75),
    "MEDIUM": CompoundParams(0.025, 0.00035, 1.60, 0.28, 90, 7.50, 100, 110, 80, 20, 0.92, 1.60),
    "HARD": CompoundParams(0.015, 0.00018, 1.45, 0.22, 100, 8.75, 110, 120, 85, 30, 0.94, 1.45),
    "INTERMEDIATE": CompoundParams(0.040, 0.00060, 2.00, 0.30, 65, 5.00, 70, 75, 55, 16, 0.88, 2.00),
    "WET": CompoundParams(0.035, 0.00045, 2.20, 0.28, 55, 5.00, 60, 65, 45, 18, 0.88, 2.20),
}

AGE_WEAR_LAMBDA = 0.55     # blend weight between age-based and wear-based degradation
PHI_MIN = 0.55             # minimum thermal-shape factor (grip never zeroes out from temperature alone)
GAMMA_HOT = 0.04           # overheat penalty, s per degree over t_hot
GAMMA_COLD = 0.03          # cold penalty, s per degree under t_cold
THERMAL_RATE_BOOST = 0.02  # extra degradation rate, s/lap, while running hot


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _compound_params(name: str | None) -> CompoundParams:
    key = (name or "MEDIUM").upper().replace("INTER", "INTERMEDIATE")
    return _COMPOUND_PARAMS.get(key, _COMPOUND_PARAMS["MEDIUM"])


@dataclass
class TireResult:
    grip_factor: float            # 0-1, current grip relative to a fresh tyre at ideal temperature
    health_score: float           # 0-100, blended health indicator for display
    deg_rate_s_per_lap: float     # expected extra lap-time cost next lap, s
    remaining_laps: float         # laps of competitive life left on this set
    performance_loss_s: float     # cumulative lap-time cost versus a fresh tyre, s
    overheat_risk: OverheatRisk
    compound: str
    age_laps: int
    limiting_factor: str          # which budget runs out first: wear / age / performance


def evaluate_tire(compound: str | None, age_laps: float | None, tyres_wear: tuple | None,
                  tyres_surface_temp: tuple | None) -> TireResult:
    """Score one tyre set's current grip, health and remaining competitive life.

    ``tyres_wear`` is (RL, RR, FL, FR) percent as the game reports it; ``tyres_surface_temp``
    is the same order in Celsius. Either can be None (not yet received for this car), in which
    case wear is estimated from age and the thermal term is skipped.
    """
    params = _compound_params(compound)
    age = max(0.0, float(age_laps or 0))

    if tyres_wear:
        wear_frac = _clamp(sum(tyres_wear) / 400.0, 0.0, 1.0)   # average of 4 wheels, 0-100% each
    else:
        wear_frac = _clamp(age / max(params.expected_life_laps, 1.0), 0.0, 1.0)

    temps = [t for t in (tyres_surface_temp or ()) if t is not None]
    mean_temp = sum(temps) / len(temps) if temps else None
    peak_temp = max(temps) if temps else mean_temp

    if mean_temp is not None:
        z = (mean_temp - params.t_star) / max(params.sigma, 1e-6)
        phi = _clamp(math.exp(-0.5 * z * z), PHI_MIN, 1.0)      # Gaussian thermal grip window
    else:
        phi = 1.0

    grip = _clamp((1.0 - params.a_wear * wear_frac) * phi, 0.0, 1.0)

    dt_age = params.alpha * age + params.beta * (age ** 2)
    dt_wear = params.kappa * wear_frac
    dt_thermal = 0.0
    if mean_temp is not None:
        dt_thermal += GAMMA_HOT * max(0.0, mean_temp - params.t_hot)
        dt_thermal += GAMMA_COLD * max(0.0, params.t_cold - mean_temp)
    performance_loss = AGE_WEAR_LAMBDA * dt_age + (1.0 - AGE_WEAR_LAMBDA) * dt_wear + dt_thermal

    wear_rate = wear_frac / max(age, 1.0)
    r_age = params.alpha + 2.0 * params.beta * age
    r_wear = (1.0 - AGE_WEAR_LAMBDA) * params.kappa * wear_rate
    r_therm = THERMAL_RATE_BOOST if (mean_temp is not None and mean_temp > params.t_hot) else 0.0
    deg_rate = max(0.0, r_age + r_wear + r_therm)

    eps = 1e-4
    if wear_frac >= params.wear_cliff or performance_loss >= params.competitive_loss_s:
        remaining, limiting = 0.0, "cliff"
    else:
        n_wear = (params.wear_cliff - wear_frac) / max(wear_rate, eps)
        n_age = max(0.0, params.expected_life_laps - age)
        n_perf = max(0.0, (params.competitive_loss_s - performance_loss) / max(deg_rate, eps))
        remaining = min(n_wear, n_age, n_perf)
        limiting = "wear" if remaining == n_wear else ("age" if remaining == n_age else "performance")

    health = 100.0 * _clamp(
        0.35 * grip
        + 0.25 * (1.0 - wear_frac / max(params.wear_cliff, 1e-6))
        + 0.25 * (remaining / max(params.expected_life_laps, 1e-6))
        + 0.15 * (1.0 - performance_loss / max(params.competitive_loss_s, 1e-6)),
        0.0, 1.0)

    if peak_temp is None:
        overheat = OverheatRisk.NONE
    elif peak_temp >= params.t_crit:
        overheat = OverheatRisk.CRITICAL
    elif peak_temp >= params.t_hot + 5.0:
        overheat = OverheatRisk.HIGH
    elif peak_temp >= params.t_hot:
        overheat = OverheatRisk.MEDIUM
    elif peak_temp >= params.t_hot - 5.0:
        overheat = OverheatRisk.LOW
    else:
        overheat = OverheatRisk.NONE

    return TireResult(
        grip_factor=round(grip, 4), health_score=round(health, 1),
        deg_rate_s_per_lap=round(deg_rate, 4), remaining_laps=round(remaining, 1),
        performance_loss_s=round(performance_loss, 3), overheat_risk=overheat,
        compound=(compound or "MEDIUM").upper(), age_laps=int(age), limiting_factor=limiting)


# ──────────────────────────────────────────────
# Fuel model
# ──────────────────────────────────────────────

FUEL_RESERVE_MIN_KG = 1.5
FUEL_SAVE_THRESHOLD_LAPS = 0.5
FUEL_PUSH_THRESHOLD_LAPS = 1.5


@dataclass
class FuelResult:
    fuel_kg: float
    consumption_kg_per_lap: float | None
    margin_laps: float | None      # positive = spare laps of fuel beyond the finish, with reserve held back
    can_push: bool | None
    must_save: bool | None


def evaluate_fuel(fuel_kg: float | None, consumption_kg_per_lap: float | None,
                  laps_remaining: int | None) -> FuelResult:
    """Score fuel margin from a learned per-lap burn rate (see strategy_ml.fuel_per_lap)."""
    if fuel_kg is None:
        return FuelResult(0.0, consumption_kg_per_lap, None, None, None)
    margin = can_push = must_save = None
    if consumption_kg_per_lap and consumption_kg_per_lap > 0 and laps_remaining is not None:
        margin = (fuel_kg - FUEL_RESERVE_MIN_KG) / consumption_kg_per_lap - float(laps_remaining)
        must_save = margin < FUEL_SAVE_THRESHOLD_LAPS
        can_push = margin > FUEL_PUSH_THRESHOLD_LAPS
    return FuelResult(round(fuel_kg, 2), consumption_kg_per_lap,
                      round(margin, 2) if margin is not None else None, can_push, must_save)


# ──────────────────────────────────────────────
# Opponent threat model
# ──────────────────────────────────────────────

THREAT_DRS_GAP_S = 1.0
THREAT_WATCH_GAP_S = 2.5
THREAT_GAP_MAX_SCORE = 40.0
THREAT_DRS_BONUS = 12.0
THREAT_PACE_PER_S = 55.0
THREAT_PACE_CAP = 28.0
THREAT_CATCH_LAPS_CAP = 8.0
THREAT_HIGH = 55.0
THREAT_MEDIUM = 30.0


@dataclass
class OpponentThreat:
    car_index: int
    position: int | None
    threat_score: float        # 0-100
    threat_level: str          # LOW / MEDIUM / HIGH
    gap_s: float                # signed: positive = ahead of the player, negative = behind
    pace_delta_s: float | None  # positive = this car is faster per lap than the player
    laps_to_catch: float | None
    factors: list[str]


def evaluate_opponent_threat(car_index: int, position: int | None, gap_s: float,
                             pace_delta_s: float | None) -> OpponentThreat:
    """Score how much pressure one nearby car puts on the player right now.

    ``gap_s`` follows RaceRadio's convention: positive means the car is ahead of the player
    (closing from in front matters for DRS/defending), negative means it is behind (attacking).
    """
    abs_gap = abs(gap_s)
    score = 0.0
    factors: list[str] = []

    fade = _clamp(1.0 - abs_gap / THREAT_WATCH_GAP_S, 0.0, 1.0)
    score += THREAT_GAP_MAX_SCORE * fade
    if abs_gap <= THREAT_DRS_GAP_S:
        score += THREAT_DRS_BONUS
        factors.append(f"gap {abs_gap:.1f}s (DRS range)")

    # A car behind that is also faster is closing; a car ahead that is faster is pulling away
    # (still worth tracking for the undercut, but it's not pressure on the player this lap).
    if pace_delta_s is not None and gap_s < 0 and pace_delta_s < 0:
        pts = min(THREAT_PACE_CAP, -pace_delta_s * THREAT_PACE_PER_S)
        score += pts
        if pace_delta_s <= -0.12:
            factors.append(f"faster by {-pace_delta_s:.2f}s/lap")

    laps_to_catch = None
    if pace_delta_s is not None and abs(pace_delta_s) > 1e-6:
        if gap_s < 0 and pace_delta_s < 0:        # behind and closing
            laps_to_catch = abs_gap / (-pace_delta_s)
        elif gap_s > 0 and pace_delta_s > 0:       # ahead and pulling away is not a catch
            laps_to_catch = None
    if laps_to_catch is not None and laps_to_catch <= THREAT_CATCH_LAPS_CAP:
        score += 18.0 * (1.0 - laps_to_catch / THREAT_CATCH_LAPS_CAP)
        factors.append(f"catching in ~{laps_to_catch:.1f} laps")

    score = _clamp(score, 0.0, 100.0)
    level = "HIGH" if score >= THREAT_HIGH else "MEDIUM" if score >= THREAT_MEDIUM else "LOW"

    return OpponentThreat(
        car_index=car_index, position=position, threat_score=round(score, 1), threat_level=level,
        gap_s=round(gap_s, 2), pace_delta_s=round(pace_delta_s, 3) if pace_delta_s is not None else None,
        laps_to_catch=round(laps_to_catch, 1) if laps_to_catch is not None else None, factors=factors)
