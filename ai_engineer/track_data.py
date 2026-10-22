"""
F1 2021 Track Database & Setup Advisor
======================================
Comprehensive per-track database for all 24 official circuits in F1 2021,
including post-launch DLC circuits (Imola, Portimao, Jeddah).

Contains:
  • Complete 6-component car setups:
      1. Aerodynamics (Front/Rear Wing 1-11)
      2. Transmission (On/Off Throttle Diff 50-100%)
      3. Suspension Geometry (Camber & Toe)
      4. Suspension (Suspension, Anti-Roll Bars, Ride Heights 1-11)
      5. Brakes (Pressure 80-100%, Bias 50-70%)
      6. Tyres (Pressures PSI)
  • Detailed circuit characteristics (Downforce, Tyre Wear, DRS Zones)
  • Corner-by-corner braking points and callouts
  • AI Setup Advisor prompt construction & fallback generator
"""

from typing import Any

# ──────────────────────────────────────────────
# 24 Circuits Database (F1 2021 Game)
# ──────────────────────────────────────────────

TRACK_DATABASE: dict[int, dict[str, Any]] = {
    # 0: Albert Park, Melbourne (Australia)
    0: {
        "id": 0,
        "name": "Albert Park Circuit",
        "short_name": "Albert Park",
        "location": "Melbourne",
        "country": "Australia",
        "length_m": 5303,
        "turns": 16,
        "downforce": "High",
        "tyre_wear": "Medium",
        "drs_zones": 3,
        "characteristics": "Semi-street circuit with slippery surface early on, heavy braking into Turn 1 and 3, and high traction demand.",
        "setup": {
            "aerodynamics": {"front_wing": 8, "rear_wing": 9},
            "transmission": {"diff_on_throttle": 60, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -2.70,
                "rear_camber": -1.50,
                "front_toe": 0.08,
                "rear_toe": 0.23,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 96, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.4,
                "front_left_pressure": 23.4,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Good front-end bite is essential for the rapid chicane changes. Soft rear suspension aids traction out of Turn 2 and Turn 16.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Soften front anti-roll bar to 3 to prevent understeer on standing water.",
        },
        "braking_points": [
            (280, "Turn 1 Brabham", "Heavy braking into Turn 1, right-hander"),
            (760, "Turn 3 Sports Center", "Brake hard into Turn 3 hairpin"),
            (1250, "Turn 4-5", "Short brake and flick through 4 and 5"),
            (2100, "Turn 9", "Medium braking into the Clark chicane"),
            (2950, "Turn 11-12 Waite", "High-speed chicane, slight lift and quick stab on brakes"),
            (4100, "Turn 13 Ascari", "Hard braking for Ascari"),
            (4850, "Turn 15", "Heavy braking into penultimate corner"),
        ],
    },
    # 1: Circuit Paul Ricard, Le Castellet (France)
    1: {
        "id": 1,
        "name": "Circuit Paul Ricard",
        "short_name": "Paul Ricard",
        "location": "Le Castellet",
        "country": "France",
        "length_m": 5842,
        "turns": 15,
        "downforce": "Medium",
        "tyre_wear": "Medium",
        "drs_zones": 2,
        "characteristics": "High top-speed on Mistral Straight split by chicane; technical final sector requiring front grip and balance.",
        "setup": {
            "aerodynamics": {"front_wing": 5, "rear_wing": 6},
            "transmission": {"diff_on_throttle": 65, "diff_off_throttle": 56},
            "suspension_geometry": {
                "front_camber": -2.80,
                "rear_camber": -1.40,
                "front_toe": 0.07,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 5,
                "rear_suspension": 4,
                "front_anti_roll": 7,
                "rear_anti_roll": 5,
                "front_ride_height": 2,
                "rear_ride_height": 4,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 57},
            "tyres": {
                "front_right_pressure": 23.8,
                "front_left_pressure": 23.8,
                "rear_right_pressure": 22.2,
                "rear_left_pressure": 22.2,
            },
            "notes": "Low drag setup preserves overtaking potential down Mistral while stiff front ARB controls roll in Sector 3.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+1",
            "brake_bias": "55%",
            "notes": "Keep diff open (54%) to prevent snap oversteer on the kerbs.",
        },
        "braking_points": [
            (320, "Turn 1 S de la Verrerie", "Hard braking into Turn 1 chicane"),
            (1150, "Turn 3-4 Sainte-Baume", "Trail brake into Sainte-Baume"),
            (2500, "Mistral Chicane", "Heavy stop from 330 km/h into Mistral chicane"),
            (3700, "Signes", "Flat out or gentle lift into Signes"),
            (4200, "Le Beausset", "Double apex right-hander, modulate trail braking"),
            (5200, "Bendor & Pont", "Slow technical final complex, brake early for rotation"),
        ],
    },
    # 2: Shanghai International Circuit (China)
    2: {
        "id": 2,
        "name": "Shanghai International Circuit",
        "short_name": "Shanghai",
        "location": "Shanghai",
        "country": "China",
        "length_m": 5451,
        "turns": 16,
        "downforce": "Medium",
        "tyre_wear": "High",
        "drs_zones": 2,
        "characteristics": "Front-left tyre killer through Turn 1-4 snail complex; massive 1.2km back straight requires low aerodynamic drag.",
        "setup": {
            "aerodynamics": {"front_wing": 5, "rear_wing": 6},
            "transmission": {"diff_on_throttle": 62, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -3.00,
                "rear_camber": -1.60,
                "front_toe": 0.08,
                "rear_toe": 0.24,
            },
            "suspension": {
                "front_suspension": 5,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 97, "brake_bias": 57},
            "tyres": {
                "front_right_pressure": 23.6,
                "front_left_pressure": 24.2,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Slightly asymmetric tyre pressure compensates for heavy Front-Left load through Turn 1 and 13.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+3",
            "ride_height": "+2",
            "brake_bias": "55%",
            "notes": "Soften front suspension to navigate the downhill snail complex without sliding.",
        },
        "braking_points": [
            (250, "Turn 1-2 Snail", "Brake gently while tightening radius into Turn 1"),
            (850, "Turn 6", "Hairpin braking, 2nd gear"),
            (1600, "Turn 7-8", "High-speed left-right flick, light brake tap"),
            (2500, "Turn 11-12", "Double chicane, brake in a straight line"),
            (4150, "Turn 14 Hairpin", "Massive braking zone at end of 1.2km straight"),
            (5050, "Turn 16", "Final corner onto pit straight, smooth brake release"),
        ],
    },
    # 3: Bahrain International Circuit, Sakhir (Bahrain)
    3: {
        "id": 3,
        "name": "Bahrain International Circuit",
        "short_name": "Bahrain",
        "location": "Sakhir",
        "country": "Bahrain",
        "length_m": 5412,
        "turns": 15,
        "downforce": "Medium",
        "tyre_wear": "Very High",
        "drs_zones": 3,
        "characteristics": "Heavy traction demands, abrasive asphalt causing severe rear tyre degradation, and locking-prone Turn 10 downhill.",
        "setup": {
            "aerodynamics": {"front_wing": 6, "rear_wing": 7},
            "transmission": {"diff_on_throttle": 58, "diff_off_throttle": 52},
            "suspension_geometry": {
                "front_camber": -2.80,
                "rear_camber": -1.40,
                "front_toe": 0.06,
                "rear_toe": 0.20,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 3,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 95, "brake_bias": 55},
            "tyres": {
                "front_right_pressure": 23.2,
                "front_left_pressure": 23.2,
                "rear_right_pressure": 21.8,
                "rear_left_pressure": 21.8,
            },
            "notes": "Soft rear anti-roll bar and low on-throttle differential (58%) protect rear traction on corner exit.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Rare desert rain creates extremely greasy surface; open diff to 50%.",
        },
        "braking_points": [
            (410, "Turn 1", "Heavy braking zone, turn 1, down to 1st gear"),
            (780, "Turn 4", "Brake for turn 4, second gear"),
            (1290, "Turn 6", "Brake hard for turn 6"),
            (1670, "Turn 8", "Lift and brake for turn 8 hairpin"),
            (2050, "Turn 10", "Downhill off-camber lockup hazard, brake in straight line"),
            (2890, "Turn 11", "Uphill sweep, medium brake pressure"),
            (3500, "Turn 13", "Short braking zone, turn 13"),
            (4150, "Turn 14", "Brake for final hairpin, good exit vital for DRS straight"),
        ],
    },
    # 4: Circuit de Barcelona-Catalunya (Spain)
    4: {
        "id": 4,
        "name": "Circuit de Barcelona-Catalunya",
        "short_name": "Barcelona",
        "location": "Montmeló",
        "country": "Spain",
        "length_m": 4675,
        "turns": 16,
        "downforce": "High",
        "tyre_wear": "High",
        "drs_zones": 2,
        "characteristics": "Aero benchmark circuit; Turn 3 puts colossal sustained G on front-left, final sector chicane demands compliance.",
        "setup": {
            "aerodynamics": {"front_wing": 8, "rear_wing": 9},
            "transmission": {"diff_on_throttle": 62, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -2.90,
                "rear_camber": -1.50,
                "front_toe": 0.08,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 5,
                "rear_suspension": 4,
                "front_anti_roll": 7,
                "rear_anti_roll": 4,
                "front_ride_height": 2,
                "rear_ride_height": 4,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.5,
                "front_left_pressure": 24.0,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "High aero balance keeps the car planted through Turn 3 and Turn 9 Campsa. Stiff front ARB aids change of direction.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Raise rear ride height to clear sausage kerbs in Turn 14-15 chicane.",
        },
        "braking_points": [
            (320, "Turn 1 Elf", "Brake hard for Turn 1, chicane complex"),
            (750, "Turn 3 Renault", "Flat or gentle lift into long Turn 3 sweeper"),
            (1200, "Turn 5 Seat", "Brake downhill for Turn 5, tight left-hander"),
            (1780, "Turn 7-8", "Uphill chicane, lift and clip kerbs"),
            (2350, "Turn 9 Campsa", "High speed right-hander, small lift or tap"),
            (2950, "Turn 10 Caixa", "Heavy braking into newly configured Turn 10"),
            (3700, "Turn 12", "Trail brake into long right-hander"),
            (4200, "Turn 14-15 Chicane", "Slowest part of circuit, aggressive kerb hopping"),
        ],
    },
    # 5: Circuit de Monaco (Monaco)
    5: {
        "id": 5,
        "name": "Circuit de Monaco",
        "short_name": "Monaco",
        "location": "Monte Carlo",
        "country": "Monaco",
        "length_m": 3337,
        "turns": 19,
        "downforce": "Maximum",
        "tyre_wear": "Low",
        "drs_zones": 1,
        "characteristics": "Ultimate street circuit with zero room for error; tightest hairpin in motorsport, maximum steering lock and downforce.",
        "setup": {
            "aerodynamics": {"front_wing": 11, "rear_wing": 11},
            "transmission": {"diff_on_throttle": 52, "diff_off_throttle": 50},
            "suspension_geometry": {
                "front_camber": -3.50,
                "rear_camber": -2.00,
                "front_toe": 0.12,
                "rear_toe": 0.20,
            },
            "suspension": {
                "front_suspension": 2,
                "rear_suspension": 2,
                "front_anti_roll": 3,
                "rear_anti_roll": 2,
                "front_ride_height": 4,
                "rear_ride_height": 6,
            },
            "brakes": {"brake_pressure": 90, "brake_bias": 53},
            "tyres": {
                "front_right_pressure": 22.0,
                "front_left_pressure": 22.0,
                "rear_right_pressure": 20.5,
                "rear_left_pressure": 20.5,
            },
            "notes": "Maximum wings, ultra-soft suspension to absorb bumps and manhole covers, and unlocked diff for razor-sharp rotation.",
        },
        "wet_adjustments": {
            "front_wing": "+0 (Already 11)",
            "rear_wing": "+0",
            "ride_height": "+3",
            "brake_bias": "52%",
            "notes": "Monaco in the wet is treacherous; lower tyre pressures to minimum to build temperature.",
        },
        "braking_points": [
            (180, "Sainte Dévote", "Brake for Turn 1 Sainte Devote, watch inside kerb"),
            (720, "Massenet", "High speed entry into Casino Square"),
            (980, "Casino Square", "Crest of hill, brake slightly and avoid bump"),
            (1200, "Mirabeau Haute", "Downhill heavy braking into Mirabeau"),
            (1380, "Grand Hotel Hairpin", "Full steering lock, 1st gear, slowest corner in F1"),
            (1750, "Mirabeau Bas / Portier", "Careful on throttle onto tunnel run"),
            (2050, "Nouvelle Chicane", "Heavy stop coming out of tunnel, 200m board"),
            (2700, "Tabac", "Commit in 4th gear, skim the armco barrier"),
            (2950, "Swimming Pool", "High-speed chicane, ride first kerb, avoid second"),
            (3250, "La Rascasse", "Tight hairpin into Anthony Noghes and finish straight"),
        ],
    },
    # 6: Circuit Gilles Villeneuve, Montreal (Canada)
    6: {
        "id": 6,
        "name": "Circuit Gilles Villeneuve",
        "short_name": "Canada",
        "location": "Montreal",
        "country": "Canada",
        "length_m": 4361,
        "turns": 14,
        "downforce": "Low-Medium",
        "tyre_wear": "Medium",
        "drs_zones": 3,
        "characteristics": "Stop-and-go circuit with high kerb demands, heavy braking zones, and the infamous Wall of Champions.",
        "setup": {
            "aerodynamics": {"front_wing": 4, "rear_wing": 5},
            "transmission": {"diff_on_throttle": 64, "diff_off_throttle": 56},
            "suspension_geometry": {
                "front_camber": -2.80,
                "rear_camber": -1.40,
                "front_toe": 0.07,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 3,
                "rear_suspension": 3,
                "front_anti_roll": 5,
                "rear_anti_roll": 4,
                "front_ride_height": 4,
                "rear_ride_height": 6,
            },
            "brakes": {"brake_pressure": 100, "brake_bias": 57},
            "tyres": {
                "front_right_pressure": 23.4,
                "front_left_pressure": 23.4,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Low downforce for high top speed, compliant ride height (4/6) to ride the chicanes without launching the chassis.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "55%",
            "notes": "Smooth braking into the final chicane to avoid aquaplaning into the Wall of Champions.",
        },
        "braking_points": [
            (350, "Turn 1-2 Senna", "Brake into Senna 'S', left then long right"),
            (900, "Turn 3-4", "Quick chicane, jump the kerbs"),
            (1500, "Turn 6-7", "Hard brake for left-right chicane"),
            (1900, "Turn 8-9", "Approach under bridge, brake in a straight line"),
            (3100, "Turn 10 L'Epingle Hairpin", "Heavy braking into the hairpin, 1st or 2nd gear"),
            (4100, "Turn 13-14 Wall of Champions", "Massive braking from 335 km/h, commit to chicane kerbs"),
        ],
    },
    # 7: Silverstone Circuit (Great Britain)
    7: {
        "id": 7,
        "name": "Silverstone Circuit",
        "short_name": "Silverstone",
        "location": "Northamptonshire",
        "country": "Great Britain",
        "length_m": 5891,
        "turns": 18,
        "downforce": "Medium-High",
        "tyre_wear": "Very High",
        "drs_zones": 2,
        "characteristics": "High-speed temple of aero loads; Maggots-Becketts-Chapel produces the highest sustained lateral acceleration on the calendar.",
        "setup": {
            "aerodynamics": {"front_wing": 7, "rear_wing": 8},
            "transmission": {"diff_on_throttle": 68, "diff_off_throttle": 56},
            "suspension_geometry": {
                "front_camber": -3.20,
                "rear_camber": -1.60,
                "front_toe": 0.08,
                "rear_toe": 0.24,
            },
            "suspension": {
                "front_suspension": 6,
                "rear_suspension": 5,
                "front_anti_roll": 8,
                "rear_anti_roll": 6,
                "front_ride_height": 2,
                "rear_ride_height": 4,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.8,
                "front_left_pressure": 24.4,
                "rear_right_pressure": 22.4,
                "rear_left_pressure": 22.4,
            },
            "notes": "Stiff suspension controls aerodynamic platform height at 290+ km/h through Copse and Maggots.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "British weather changes fast; keep rear wing high to prevent snap spins through Stowe.",
        },
        "braking_points": [
            (290, "Turn 1 Abbey", "Flat out or gentle lift into Abbey"),
            (850, "Turn 3 Village", "Lift and hard braking for Village chicane"),
            (1180, "Turn 4 The Loop", "Slowest corner, tight left hairpin"),
            (1680, "Turn 6 Brooklands", "Heavy braking into Brooklands"),
            (2050, "Turn 7 Luffield", "Long sweeping right, maintain momentum"),
            (2600, "Turn 9 Copse", "Terrifying 280 km/h entry, small confidence lift"),
            (3400, "Maggots-Becketts", "Dance of high-speed direction changes, precision throttle"),
            (4250, "Turn 14 Chapel", "Flat out onto Hangar Straight"),
            (4950, "Turn 15 Stowe", "Heavy braking into Stowe from 320 km/h"),
            (5400, "Turn 16 Vale", "Hard brake for Vale chicane, then accelerate through Club"),
        ],
    },
    # 9: Hungaroring (Hungary)
    9: {
        "id": 9,
        "name": "Hungaroring",
        "short_name": "Hungaroring",
        "location": "Mogyoród",
        "country": "Hungary",
        "length_m": 4381,
        "turns": 14,
        "downforce": "Maximum",
        "tyre_wear": "High",
        "drs_zones": 2,
        "characteristics": "Monaco without the walls; relentless twisty layout with no rest for driver or tyres in searing summer heat.",
        "setup": {
            "aerodynamics": {"front_wing": 10, "rear_wing": 11},
            "transmission": {"diff_on_throttle": 56, "diff_off_throttle": 52},
            "suspension_geometry": {
                "front_camber": -3.20,
                "rear_camber": -1.80,
                "front_toe": 0.09,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 95, "brake_bias": 55},
            "tyres": {
                "front_right_pressure": 23.0,
                "front_left_pressure": 23.0,
                "rear_right_pressure": 21.5,
                "rear_left_pressure": 21.5,
            },
            "notes": "Maximum downforce is essential. Responsive front end needed for constant mid-corner rotation.",
        },
        "wet_adjustments": {
            "front_wing": "+0",
            "rear_wing": "+0",
            "ride_height": "+2",
            "brake_bias": "53%",
            "notes": "Off-camber corners become ice rinks in the wet; soften rear anti-roll bar to 2.",
        },
        "braking_points": [
            (300, "Turn 1", "Heavy downhill braking zone, main overtaking spot"),
            (750, "Turn 2", "Downhill off-camber left-hander, easy to slide wide"),
            (1200, "Turn 4", "Blind crest uphill left, high-speed commitment"),
            (1650, "Turn 6-7 Chicane", "Heavy stop for the chicane"),
            (2200, "Turn 8-9", "Flowing mid-speed corners, keep rhythm"),
            (3200, "Turn 11", "Fast right-hander, clip kerb cleanly"),
            (3700, "Turn 12", "90-degree right, precise braking"),
            (4050, "Turn 13-14", "Long penultimate carousel, smooth exit for pit straight"),
        ],
    },
    # 10: Circuit de Spa-Francorchamps (Belgium)
    10: {
        "id": 10,
        "name": "Circuit de Spa-Francorchamps",
        "short_name": "Spa",
        "location": "Stavelot",
        "country": "Belgium",
        "length_m": 7004,
        "turns": 19,
        "downforce": "Medium-Low",
        "tyre_wear": "High",
        "drs_zones": 2,
        "characteristics": "The rollercoaster of the Ardennes; Eau Rouge/Raidillon compression, Kemmel Straight speed vs Sector 2 downforce compromise.",
        "setup": {
            "aerodynamics": {"front_wing": 4, "rear_wing": 5},
            "transmission": {"diff_on_throttle": 66, "diff_off_throttle": 56},
            "suspension_geometry": {
                "front_camber": -3.00,
                "rear_camber": -1.50,
                "front_toe": 0.08,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 5,
                "rear_suspension": 4,
                "front_anti_roll": 7,
                "rear_anti_roll": 5,
                "front_ride_height": 2,
                "rear_ride_height": 4,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.6,
                "front_left_pressure": 24.2,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Low wing gives 335+ km/h onto Kemmel. Avoid bottoming out in Raidillon dip with sufficient ride height.",
        },
        "wet_adjustments": {
            "front_wing": "+3",
            "rear_wing": "+3",
            "ride_height": "+3",
            "brake_bias": "55%",
            "notes": "Micro-climates often wet Sector 2 while Sector 1 is bone dry; aero trim gives confidence in wet Pouhon.",
        },
        "braking_points": [
            (250, "Turn 1 La Source", "Heavy downhill braking for La Source hairpin"),
            (1100, "Eau Rouge / Raidillon", "Hold breath, flat out compression up to Kemmel"),
            (1800, "Les Combes", "Heavy braking from 335 km/h, chicane overtaking spot"),
            (2400, "Turn 8 Bruxelles", "Long downhill right-hander, feather throttle"),
            (2900, "Turn 9 Speaker's Corner", "Off-camber left-hander, easy to lose rear"),
            (4100, "Turn 10-11 Pouhon", "Double apex left, 260 km/h entry, tiny lift"),
            (4750, "Turn 12-13 Fagnes", "Quick flick chicane, jump curbs cleanly"),
            (5200, "Turn 14-15 Stavelot", "Brake into right, carry speed through Paul Frere"),
            (5800, "Blanchimont", "Terrifying flat out 300+ km/h sweep"),
            (6600, "Bus Stop Chicane", "Huge stop into final chicane, great passing zone"),
        ],
    },
    # 11: Autodromo Nazionale Monza (Italy)
    11: {
        "id": 11,
        "name": "Autodromo Nazionale Monza",
        "short_name": "Monza",
        "location": "Monza",
        "country": "Italy",
        "length_m": 5793,
        "turns": 11,
        "downforce": "Low",
        "tyre_wear": "Low",
        "drs_zones": 2,
        "characteristics": "The Temple of Speed; 80% full throttle, skinny wings, 350+ km/h top speeds, and violent braking into chicanes.",
        "setup": {
            "aerodynamics": {"front_wing": 1, "rear_wing": 2},
            "transmission": {"diff_on_throttle": 70, "diff_off_throttle": 58},
            "suspension_geometry": {
                "front_camber": -2.60,
                "rear_camber": -1.20,
                "front_toe": 0.06,
                "rear_toe": 0.20,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 7,
                "rear_anti_roll": 4,
                "front_ride_height": 2,
                "rear_ride_height": 3,
            },
            "brakes": {"brake_pressure": 100, "brake_bias": 58},
            "tyres": {
                "front_right_pressure": 24.2,
                "front_left_pressure": 24.2,
                "rear_right_pressure": 22.8,
                "rear_left_pressure": 22.8,
            },
            "notes": "Skinny wings. Maximum brake pressure (100%) to decelerate from 355 km/h into Variante del Rettifilo.",
        },
        "wet_adjustments": {
            "front_wing": "+3",
            "rear_wing": "+3",
            "ride_height": "+2",
            "brake_bias": "55%",
            "notes": "Low downforce on wet Monza is deadly; increase wings to 4/5.",
        },
        "braking_points": [
            (
                340,
                "Variante del Rettifilo (Turn 1-2)",
                "Brutal braking from 355 km/h to 70 km/h, 1st chicane",
            ),
            (1230, "Curva Grande (Turn 3)", "Flat out acceleration sweep"),
            (1950, "Variante della Roggia (Turn 4-5)", "Heavy braking from 330 km/h, second chicane"),
            (3550, "Lesmo 1 (Turn 6)", "Brake hard into 4th gear, commit early"),
            (3950, "Lesmo 2 (Turn 7)", "3rd gear right, clip apex and use all kerb"),
            (4700, "Variante Ascari (Turn 8-10)", "Fast chicane, brake at 100m, flick left-right-left"),
            (5350, "Curva Parabolica (Turn 11)", "Long right-hander, brake at 80m, open steering on exit"),
        ],
    },
    # 12: Marina Bay Street Circuit (Singapore)
    12: {
        "id": 12,
        "name": "Marina Bay Street Circuit",
        "short_name": "Marina Bay",
        "aliases": ["Singapore", "Marina Bay"],
        "location": "Marina Bay",
        "country": "Singapore",
        "length_m": 5063,
        "turns": 23,
        "downforce": "Maximum",
        "tyre_wear": "High",
        "drs_zones": 3,
        "characteristics": "Bumpy night race with oppressive humidity; 23 corners leave zero breathing room, punishing traction out of slow turns.",
        "setup": {
            "aerodynamics": {"front_wing": 10, "rear_wing": 11},
            "transmission": {"diff_on_throttle": 54, "diff_off_throttle": 52},
            "suspension_geometry": {
                "front_camber": -3.30,
                "rear_camber": -1.80,
                "front_toe": 0.10,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 3,
                "rear_suspension": 2,
                "front_anti_roll": 4,
                "rear_anti_roll": 3,
                "front_ride_height": 4,
                "rear_ride_height": 6,
            },
            "brakes": {"brake_pressure": 94, "brake_bias": 55},
            "tyres": {
                "front_right_pressure": 22.8,
                "front_left_pressure": 22.8,
                "rear_right_pressure": 21.2,
                "rear_left_pressure": 21.2,
            },
            "notes": "Soft springs absorb the bumps across the Anderson Bridge. High wings keep rear planted under braking.",
        },
        "wet_adjustments": {
            "front_wing": "+0",
            "rear_wing": "+0",
            "ride_height": "+2",
            "brake_bias": "53%",
            "notes": "Street lines and painted kerbs are lethal in the wet; avoid throttle over paint.",
        },
        "braking_points": [
            (350, "Turn 1-2-3", "Heavy braking into Turn 1 left, flick through 2 and 3"),
            (700, "Turn 5", "Short brake into right sweep"),
            (1400, "Turn 7", "Brake at end of Raffles Blvd DRS straight"),
            (1900, "Turn 8", "90-degree right onto St. Andrews Road"),
            (2300, "Turn 10 Singapore Sling", "Left-hander, watch outside barrier"),
            (2700, "Turn 11-12", "Technical chicanes under Anderson Bridge"),
            (3200, "Turn 13 Hairpin", "Heavy stop into tight hairpin"),
            (4300, "Turn 16-17", "Flick through the grandstand section"),
            (4650, "Turn 18-19 Under The Grandstand", "Tunnel-like 90 degree corners"),
            (5000, "Turn 20-21", "Final chicane leading to finish straight"),
        ],
    },
    # 13: Suzuka International Racing Course (Japan)
    13: {
        "id": 13,
        "name": "Suzuka International Racing Course",
        "short_name": "Suzuka",
        "location": "Suzuka",
        "country": "Japan",
        "length_m": 5807,
        "turns": 18,
        "downforce": "High",
        "tyre_wear": "Very High",
        "drs_zones": 1,
        "characteristics": "The driver's ultimate circuit; figure-8 layout featuring the legendary Sector 1 Esses, Degners, Spoon, and 130R.",
        "setup": {
            "aerodynamics": {"front_wing": 8, "rear_wing": 9},
            "transmission": {"diff_on_throttle": 64, "diff_off_throttle": 56},
            "suspension_geometry": {
                "front_camber": -3.20,
                "rear_camber": -1.60,
                "front_toe": 0.08,
                "rear_toe": 0.24,
            },
            "suspension": {
                "front_suspension": 6,
                "rear_suspension": 5,
                "front_anti_roll": 8,
                "rear_anti_roll": 6,
                "front_ride_height": 2,
                "rear_ride_height": 4,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 24.2,
                "front_left_pressure": 23.8,
                "rear_right_pressure": 22.2,
                "rear_left_pressure": 22.2,
            },
            "notes": "Razor sharp front-end responsiveness is required through the Esses. Oversteering in Degner 2 or 130R means heavy crash.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+3",
            "brake_bias": "54%",
            "notes": "Rivers frequently form across the track in Esses; lift through standing water.",
        },
        "braking_points": [
            (300, "Turn 1-2", "High speed entry in 6th, down to 3rd for Turn 2"),
            (1050, "Turn 3-6 The Esses", "Rhythm section, progressive lifts, maintain momentum"),
            (1700, "Turn 7 Dunlop Curve", "Long uphill left sweep, manage tyre scrub"),
            (2300, "Turn 8 Degner 1", "Fast blind right, small brake dab in 6th"),
            (2700, "Turn 9 Degner 2", "Heavy stop over yellow kerb into 3rd gear"),
            (3100, "Turn 11 Hairpin", "Full braking down to 1st gear, tricky camber"),
            (3900, "Turn 13-14 Spoon Curve", "Double apex left, trail brake into 4th gear"),
            (4800, "Turn 15 130R", "Flat out 300 km/h left-hand sweep"),
            (5300, "Turn 16-17 Casio Triangle", "Heavy braking into the final chicane"),
        ],
    },
    # 14: Yas Marina Circuit, Abu Dhabi (UAE)
    14: {
        "id": 14,
        "name": "Yas Marina Circuit",
        "short_name": "Yas Marina",
        "aliases": ["Abu Dhabi", "Yas Marina"],
        "location": "Yas Island",
        "country": "UAE",
        "length_m": 5281,
        "turns": 16,
        "downforce": "Medium",
        "tyre_wear": "Medium",
        "drs_zones": 2,
        "characteristics": "Twilight season finale with long straights bookended by chicanes and hotel complex requiring rear traction.",
        "setup": {
            "aerodynamics": {"front_wing": 6, "rear_wing": 7},
            "transmission": {"diff_on_throttle": 62, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -2.90,
                "rear_camber": -1.50,
                "front_toe": 0.07,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 4,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.4,
                "front_left_pressure": 23.4,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Balanced setup enables strong top speed on back straights while preserving rear grip under the Yas Viceroy hotel.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Rain in Abu Dhabi is rare; keep diff unlocked to avoid snaps on painted runoff.",
        },
        "braking_points": [
            (350, "Turn 1", "Fast 4th gear left-hander off pit straight"),
            (850, "Turn 5 Chicane & Hairpin", "Hard braking into modified hairpin section"),
            (2100, "Turn 6-7", "Heavy stop from 335 km/h at end of long DRS straight"),
            (2800, "Turn 9 Banked Corner", "Carry speed around the new sweeping banked turn"),
            (3500, "Turn 11", "90-degree right, precise braking"),
            (4300, "Turn 12-14 Hotel Complex", "Blind technical rights under hotel bridge"),
            (4900, "Turn 15-16", "Double right-handers onto the start-finish straight"),
        ],
    },
    # 15: Circuit of the Americas (COTA, USA)
    15: {
        "id": 15,
        "name": "Circuit of the Americas",
        "short_name": "COTA",
        "location": "Austin",
        "country": "USA",
        "length_m": 5513,
        "turns": 20,
        "downforce": "High",
        "tyre_wear": "High",
        "drs_zones": 2,
        "characteristics": "Steep 40-meter uphill climb into blind Turn 1; rapid Sector 1 esses inspired by Silverstone, and triple-apex stadium carousel.",
        "setup": {
            "aerodynamics": {"front_wing": 8, "rear_wing": 9},
            "transmission": {"diff_on_throttle": 62, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -3.10,
                "rear_camber": -1.50,
                "front_toe": 0.08,
                "rear_toe": 0.23,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 4,
                "front_anti_roll": 7,
                "rear_anti_roll": 5,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.6,
                "front_left_pressure": 24.0,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Aero balance through the Sector 1 esses is paramount. Ride height must absorb the notorious Austin surface bumps.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+3",
            "brake_bias": "54%",
            "notes": "Bumps generate puddles; raise ride height to 5/8 to prevent high-speed aquaplaning.",
        },
        "braking_points": [
            (350, "Turn 1", "Brutal uphill braking into blind hairpin crest"),
            (1050, "Turn 3-6 The Esses", "Rapid direction changes, precision throttle modulation"),
            (1600, "Turn 7-9", "Downhill sweeps, clip apex kerbs"),
            (2000, "Turn 11 Hairpin", "Heavy stop onto back straight, critical traction"),
            (3200, "Turn 12", "Massive braking from 330 km/h, primary overtaking zone"),
            (4100, "Turn 13-15", "Technical stadium chicane"),
            (4900, "Turn 16-18 Carousel", "Multi-apex right-hander, hold car balanced on throttle"),
            (5200, "Turn 19-20", "Fast final corners before front straight"),
        ],
    },
    # 16: Autódromo José Carlos Pace, Interlagos (Brazil)
    16: {
        "id": 16,
        "name": "Autódromo José Carlos Pace",
        "short_name": "Interlagos",
        "location": "São Paulo",
        "country": "Brazil",
        "length_m": 4309,
        "turns": 15,
        "downforce": "Medium-High",
        "tyre_wear": "Medium",
        "drs_zones": 2,
        "characteristics": "Anti-clockwise bowl with steep elevation; famous Senna S downhill chicane and high-speed uphill drag to line.",
        "setup": {
            "aerodynamics": {"front_wing": 7, "rear_wing": 8},
            "transmission": {"diff_on_throttle": 62, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -3.00,
                "rear_camber": -1.50,
                "front_toe": 0.08,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 96, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.4,
                "front_left_pressure": 23.6,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Anti-clockwise track stresses right tyres. Good traction out of Junção (Turn 12) is vital for the long uphill run.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+3",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Interlagos downpours are legendary; rivers flow downhill across the Senna S.",
        },
        "braking_points": [
            (280, "Senna S (Turn 1-2)", "Downhill blind left-right, heavy braking from 330 km/h"),
            (700, "Curva do Sol (Turn 3)", "Fast uphill left onto back straight"),
            (1400, "Descida do Lago (Turn 4)", "Heavy braking zone, prime overtaking corner"),
            (2100, "Ferradura (Turn 6-7)", "Fast double-apex uphill right"),
            (2700, "Bico de Pato (Turn 8)", "Tight right hairpin, lowest gear"),
            (3200, "Mergulho (Turn 11)", "High-speed downhill compression"),
            (3900, "Junção (Turn 12)", "Crucial slow left, nail throttle for long uphill flat run"),
        ],
    },
    # 17: Red Bull Ring (Austria)
    17: {
        "id": 17,
        "name": "Red Bull Ring",
        "short_name": "Red Bull Ring",
        "location": "Spielberg",
        "country": "Austria",
        "length_m": 4318,
        "turns": 10,
        "downforce": "Low-Medium",
        "tyre_wear": "Medium",
        "drs_zones": 3,
        "characteristics": "Short, rapid sprint with severe uphill elevation into Turn 3; aggressive sausage kerbs that can break suspensions.",
        "setup": {
            "aerodynamics": {"front_wing": 4, "rear_wing": 5},
            "transmission": {"diff_on_throttle": 64, "diff_off_throttle": 56},
            "suspension_geometry": {
                "front_camber": -2.80,
                "rear_camber": -1.30,
                "front_toe": 0.06,
                "rear_toe": 0.20,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 57},
            "tyres": {
                "front_right_pressure": 23.8,
                "front_left_pressure": 23.8,
                "rear_right_pressure": 22.2,
                "rear_left_pressure": 22.2,
            },
            "notes": "Low drag maximizes overtaking with 3 consecutive DRS zones. Soft rear springs help traction out of Turn 3.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "55%",
            "notes": "Stay off yellow sausage kerbs on corner exits in wet conditions.",
        },
        "braking_points": [
            (300, "Turn 1 Niki Lauda", "Uphill 90-degree right, brake at 70m board"),
            (850, "Turn 2", "Kink, flat out"),
            (1450, "Turn 3 Remus", "Steep uphill heavy braking to 1st gear, best passing spot"),
            (2200, "Turn 4 Schlossgold", "Downhill right, easy to lock inside front wheel"),
            (3450, "Turn 6-7 Rauch", "Double fast lefts, commit to apex speed"),
            (3900, "Turn 9-10 Jochen Rindt", "High-speed rights, watch track limits on exit"),
        ],
    },
    # 18: Sochi Autodrom (Russia)
    18: {
        "id": 18,
        "name": "Sochi Autodrom",
        "short_name": "Sochi",
        "location": "Sochi",
        "country": "Russia",
        "length_m": 5848,
        "turns": 18,
        "downforce": "Medium",
        "tyre_wear": "Low",
        "drs_zones": 2,
        "characteristics": "Smooth Olympic park tarmac with 90-degree corners and sustained lateral load around the long Turn 3 omega curve.",
        "setup": {
            "aerodynamics": {"front_wing": 5, "rear_wing": 6},
            "transmission": {"diff_on_throttle": 60, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -2.80,
                "rear_camber": -1.40,
                "front_toe": 0.07,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 4,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 2,
                "rear_ride_height": 4,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 24.0,
                "front_left_pressure": 24.0,
                "rear_right_pressure": 22.4,
                "rear_left_pressure": 22.4,
            },
            "notes": "Smooth asphalt offers low tyre degradation. Front-right tyre requires temperature maintenance for Turn 3.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Sochi's slick asphalt offers near zero grip in rain; trail brake gently.",
        },
        "braking_points": [
            (450, "Turn 2", "Massive braking from 335 km/h, 90-degree right overtaking spot"),
            (1300, "Turn 3 Omega", "Endless sweeping left around medals plaza, manage front-right"),
            (1900, "Turn 4", "90-degree right, 3rd gear"),
            (2800, "Turn 7", "Brake in a straight line for technical right-left"),
            (3400, "Turn 10", "Fast blind right"),
            (4600, "Turn 13", "Heavy stop from DRS back straight"),
            (5200, "Turn 14-15", "Technical chicane around barriers"),
            (5600, "Turn 16-18", "Sequential 90-degree rights leading to pit straight"),
        ],
    },
    # 19: Autódromo Hermanos Rodríguez (Mexico)
    19: {
        "id": 19,
        "name": "Autódromo Hermanos Rodríguez",
        "short_name": "Mexico",
        "aliases": ["Mexico City", "Mexico"],
        "location": "Mexico City",
        "country": "Mexico",
        "length_m": 4304,
        "turns": 17,
        "downforce": "High (Thin Air)",
        "tyre_wear": "Low",
        "drs_zones": 3,
        "characteristics": "Altitude 2,200m above sea level creates thin air; run Monaco-wings to achieve Monza-drag levels; slow stadium section.",
        "setup": {
            "aerodynamics": {"front_wing": 9, "rear_wing": 10},
            "transmission": {"diff_on_throttle": 62, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -2.80,
                "rear_camber": -1.40,
                "front_toe": 0.07,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 57},
            "tyres": {
                "front_right_pressure": 23.4,
                "front_left_pressure": 23.4,
                "rear_right_pressure": 21.8,
                "rear_left_pressure": 21.8,
            },
            "notes": "Thin air cuts cooling and aero downforce by 25%. Run near-maximum wing while still reaching 350 km/h on main straight.",
        },
        "wet_adjustments": {
            "front_wing": "+1",
            "rear_wing": "+1",
            "ride_height": "+2",
            "brake_bias": "55%",
            "notes": "Low density air reduces rain clearing; keep wipers / visor tear-offs ready.",
        },
        "braking_points": [
            (350, "Turn 1-3 Chicane", "Huge 360 km/h braking zone into tight chicane"),
            (800, "Turn 4-5", "Brake for Turn 4 chicane"),
            (1400, "Turn 6", "Hairpin-like right, gentle throttle on exit"),
            (1900, "Turn 7-11 Esses", "Fast technical esses, balance car through direction switches"),
            (3500, "Turn 12 Peraltada Entry", "Heavy brake into baseball stadium entrance"),
            (4050, "Turn 13-16 Foro Sol", "Stadium stadium stadium! 1st gear tight slow complex"),
        ],
    },
    # 20: Baku City Circuit (Azerbaijan)
    20: {
        "id": 20,
        "name": "Baku City Circuit",
        "short_name": "Baku",
        "location": "Baku",
        "country": "Azerbaijan",
        "length_m": 6003,
        "turns": 20,
        "downforce": "Low-Medium",
        "tyre_wear": "Medium",
        "drs_zones": 2,
        "characteristics": "Extreme contrasts: 2.2km flat-out blast reaching 360 km/h alongside the claustrophobic 7.6m castle section.",
        "setup": {
            "aerodynamics": {"front_wing": 3, "rear_wing": 4},
            "transmission": {"diff_on_throttle": 60, "diff_off_throttle": 52},
            "suspension_geometry": {
                "front_camber": -2.70,
                "rear_camber": -1.30,
                "front_toe": 0.06,
                "rear_toe": 0.20,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 100, "brake_bias": 57},
            "tyres": {
                "front_right_pressure": 24.2,
                "front_left_pressure": 24.2,
                "rear_right_pressure": 22.4,
                "rear_left_pressure": 22.4,
            },
            "notes": "Low wings are required to defend on the massive 2.2km front straight. Good mechanical grip required through castle.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+3",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Baku in rain is chaotic; street paint becomes slippery glass.",
        },
        "braking_points": [
            (450, "Turn 1", "350 km/h braking into 90-degree left, major overtake spot"),
            (1100, "Turn 3", "90-degree left, brake hard at 100m"),
            (2700, "Turn 7", "Tight 90-degree right before castle entrance"),
            (3200, "Turn 8-10 Castle Section", "Narrowest track width in F1 (7.6m), thread the needle"),
            (4150, "Turn 12", "Fast downhill sweep"),
            (4600, "Turn 15", "Heavy braking downhill, notorious lockup / barrier spot"),
            (5400, "Turn 16", "Final 90-degree left, exit cleanly onto 2.2km flat out straight"),
        ],
    },
    # 26: Circuit Zandvoort (Netherlands)
    26: {
        "id": 26,
        "name": "Circuit Zandvoort",
        "short_name": "Zandvoort",
        "location": "Zandvoort",
        "country": "Netherlands",
        "length_m": 4259,
        "turns": 14,
        "downforce": "High",
        "tyre_wear": "High",
        "drs_zones": 2,
        "characteristics": "Rollercoaster in the sand dunes with 18-degree banking at Hugenholtz and Arie Luyendyk corners; high vertical tyre loads.",
        "setup": {
            "aerodynamics": {"front_wing": 9, "rear_wing": 10},
            "transmission": {"diff_on_throttle": 62, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -3.20,
                "rear_camber": -1.70,
                "front_toe": 0.08,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 5,
                "rear_suspension": 4,
                "front_anti_roll": 7,
                "rear_anti_roll": 5,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 24.2,
                "front_left_pressure": 24.6,
                "rear_right_pressure": 22.8,
                "rear_left_pressure": 22.8,
            },
            "notes": "Higher tyre pressures counter the extreme compression on the 18-degree banking. Front wing keeps nose pinned in Tarzan.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Banking drains water to the bottom; high lines around Hugenholtz are fast in wet.",
        },
        "braking_points": [
            (300, "Tarzanbocht (Turn 1)", "Heavy braking into famous cambered hairpin"),
            (
                900,
                "Gerlachbocht (Turn 2) & Hugenholtz (Turn 3)",
                "Downhill into steep 18-degree banking, take high line",
            ),
            (1500, "Turn 7 Scheivlak", "High speed blind crest right, lift slightly and trust downforce"),
            (2000, "Turn 9", "Medium brake into technical right"),
            (2700, "Turn 11-12 Hans Ernst Chicane", "Hard stop for chicane"),
            (3500, "Turn 13 Kumhobocht", "Right-hand sweep into the final banking"),
            (3900, "Turn 14 Arie Luyendyk", "Steep 18-degree banking, DRS wide open onto front straight"),
        ],
    },
    # 27: Autodromo Enzo e Dino Ferrari, Imola (Italy) [DLC]
    27: {
        "id": 27,
        "name": "Autodromo Enzo e Dino Ferrari",
        "short_name": "Imola",
        "location": "Emilia-Romagna",
        "country": "Italy",
        "length_m": 4909,
        "turns": 19,
        "downforce": "Medium-High",
        "tyre_wear": "Medium",
        "drs_zones": 1,
        "characteristics": "Historic, flowing circuit with aggressive sausage kerbs; Acque Minerali downhill and Variante Alta chicane demand compliance.",
        "setup": {
            "aerodynamics": {"front_wing": 7, "rear_wing": 8},
            "transmission": {"diff_on_throttle": 60, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -3.00,
                "rear_camber": -1.50,
                "front_toe": 0.08,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 4,
                "rear_ride_height": 6,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.4,
                "front_left_pressure": 23.8,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Compliant ride height (4/6) is vital to launch over the Variante Alta chicane without destabilizing the car.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Imola kerbs become ice in the wet; avoid greedy kerb strikes at Tamburello.",
        },
        "braking_points": [
            (300, "Tamburello (Turn 2-4)", "Brake from 325 km/h for left-right chicane"),
            (1050, "Villeneuve (Turn 5-6)", "Brake hard into chicane"),
            (1500, "Tosa (Turn 7)", "Downhill hairpin braking, 2nd gear"),
            (2100, "Piratella (Turn 9)", "Blind downhill left, small confidence brake"),
            (
                2500,
                "Acque Minerali (Turn 11-13)",
                "Tricky double right downhill, easy to run wide into gravel",
            ),
            (3400, "Variante Alta (Turn 14-15)", "Aggressive kerb riding chicane at top of hill"),
            (4100, "Rivazza 1 (Turn 17)", "Heavy downhill braking zone"),
            (4500, "Rivazza 2 (Turn 18)", "Square off exit for flat out blast to line"),
        ],
    },
    # 28: Autódromo Internacional do Algarve, Portimão (Portugal) [DLC]
    28: {
        "id": 28,
        "name": "Autódromo Internacional do Algarve",
        "short_name": "Portimao",
        "aliases": ["Portimão", "Portimao"],
        "location": "Algarve",
        "country": "Portugal",
        "length_m": 4653,
        "turns": 15,
        "downforce": "Medium-High",
        "tyre_wear": "Medium",
        "drs_zones": 2,
        "characteristics": "Modern undulating rollercoaster with blind crests and off-camber drops; Turn 1 downhill drop and sweeping final carousel.",
        "setup": {
            "aerodynamics": {"front_wing": 7, "rear_wing": 8},
            "transmission": {"diff_on_throttle": 62, "diff_off_throttle": 54},
            "suspension_geometry": {
                "front_camber": -2.90,
                "rear_camber": -1.50,
                "front_toe": 0.08,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 4,
                "rear_suspension": 3,
                "front_anti_roll": 6,
                "rear_anti_roll": 4,
                "front_ride_height": 3,
                "rear_ride_height": 5,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 56},
            "tyres": {
                "front_right_pressure": 23.6,
                "front_left_pressure": 23.6,
                "rear_right_pressure": 22.0,
                "rear_left_pressure": 22.0,
            },
            "notes": "Good front-end downforce provides confidence over blind crests. Rear stability prevents spins on the downhill plunge into Turn 1.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+2",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Dramatic elevation creates standing pools at the bottom of dips.",
        },
        "braking_points": [
            (350, "Turn 1 Primeira", "Blind crest into steep downhill right-hander"),
            (900, "Turn 3", "Tight slow hairpin, manage wheelspin on exit"),
            (1500, "Turn 5", "Downhill hairpin, lockup prone inside front"),
            (2100, "Turn 7", "High-speed sweeping right"),
            (2900, "Turn 9", "Blind uphill left crest"),
            (3400, "Turn 11 Craig Jones", "Downhill plunge into technical right"),
            (4000, "Turn 13-14", "Slow technical chicane"),
            (4350, "Turn 15 Galp", "Massive flat-out downhill right onto main straight"),
        ],
    },
    # 29: Jeddah Corniche Circuit (Saudi Arabia) [DLC]
    29: {
        "id": 29,
        "name": "Jeddah Corniche Circuit",
        "short_name": "Jeddah",
        "location": "Jeddah",
        "country": "Saudi Arabia",
        "length_m": 6174,
        "turns": 27,
        "downforce": "Low-Medium",
        "tyre_wear": "Medium",
        "drs_zones": 3,
        "characteristics": "Fastest street track in F1 history; average speeds >250 km/h between concrete walls with 3 continuous DRS zones.",
        "setup": {
            "aerodynamics": {"front_wing": 4, "rear_wing": 5},
            "transmission": {"diff_on_throttle": 64, "diff_off_throttle": 56},
            "suspension_geometry": {
                "front_camber": -2.80,
                "rear_camber": -1.40,
                "front_toe": 0.07,
                "rear_toe": 0.22,
            },
            "suspension": {
                "front_suspension": 5,
                "rear_suspension": 4,
                "front_anti_roll": 7,
                "rear_anti_roll": 5,
                "front_ride_height": 2,
                "rear_ride_height": 4,
            },
            "brakes": {"brake_pressure": 98, "brake_bias": 57},
            "tyres": {
                "front_right_pressure": 24.0,
                "front_left_pressure": 24.0,
                "rear_right_pressure": 22.4,
                "rear_left_pressure": 22.4,
            },
            "notes": "Low downforce setup for 330+ km/h speeds across 3 DRS zones. Responsive front end needed for rapid chicane changes.",
        },
        "wet_adjustments": {
            "front_wing": "+2",
            "rear_wing": "+3",
            "ride_height": "+2",
            "brake_bias": "54%",
            "notes": "Night street race with walls inches away; increased rear wing gives stability.",
        },
        "braking_points": [
            (450, "Turn 1-2", "Heavy braking into chicane from start straight"),
            (1350, "Turn 4-10 S-Curves", "Relentless 250 km/h blind sweeping curves"),
            (2500, "Turn 13 Banked Hairpin", "12-degree banked hairpin, carry speed high"),
            (3400, "Turn 16-17", "High-speed flick between concrete barriers"),
            (4200, "Turn 22-23 Chicane", "Heavy stop from 320 km/h, blind apex"),
            (5000, "Turn 25-26", "Fast sweeps leading to final corner"),
            (6000, "Turn 27", "Heavy braking for final hairpin before pit straight"),
        ],
    },
}


# ──────────────────────────────────────────────
# Public Accessors & Compatibility Interfaces
# ──────────────────────────────────────────────


def get_braking_points(track_id: int) -> list[tuple[float, str, str]]:
    """Retrieve corner braking points for a track ID."""
    circuit = TRACK_DATABASE.get(track_id)
    if circuit and "braking_points" in circuit:
        return circuit["braking_points"]
    return []


def get_setup_advice(track_id: int) -> dict[str, Any]:
    """Retrieve setup advice dictionary (backwards-compatible with legacy analysis)."""
    circuit = TRACK_DATABASE.get(track_id)
    if circuit and "setup" in circuit:
        s = circuit["setup"]
        # Convert structured setup to legacy flat format if requested
        return {
            "front_wing": f"Downforce {s['aerodynamics']['front_wing']}",
            "rear_wing": f"Downforce {s['aerodynamics']['rear_wing']}",
            "front_camber": f"{s['suspension_geometry']['front_camber']:.2f}°",
            "rear_camber": f"{s['suspension_geometry']['rear_camber']:.2f}°",
            "front_suspension": str(s['suspension']['front_suspension']),
            "rear_suspension": str(s['suspension']['rear_suspension']),
            "brake_pressure": f"{s['brakes']['brake_pressure']}%",
            "brake_bias": f"{s['brakes']['brake_bias']}%",
            "front_tyre_pressure": f"{s['tyres']['front_left_pressure']} PSI",
            "rear_tyre_pressure": f"{s['tyres']['rear_left_pressure']} PSI",
            "notes": s.get("notes", "Standard baseline setup."),
        }
    # Default fallback
    return {
        "front_wing": "Medium downforce",
        "rear_wing": "Medium downforce",
        "front_camber": "-3.00°",
        "rear_camber": "-1.50°",
        "brake_pressure": "95%",
        "brake_bias": "56%",
        "front_tyre_pressure": "23.5 PSI",
        "rear_tyre_pressure": "22.0 PSI",
        "notes": "Balanced baseline setup.",
    }


# Short/alternate layouts reuse their parent circuit's setup data (their lap distances differ,
# so braking points and corner zones are deliberately not shared).
SHORT_LAYOUT_PARENTS = {21: 3, 22: 7, 23: 15, 24: 13}


def resolve_track_id(track_id: int) -> int:
    """Database id for a game track id (short layouts map to their parent circuit)."""
    return SHORT_LAYOUT_PARENTS.get(track_id, track_id)


def corner_zones(track_id: int, track_length: int = 0) -> list[dict[str, Any]]:
    """Corner analysis windows for any circuit, built from its braking points.

    Each zone spans from just before the braking point to just after the corner, trimmed so
    neighbouring zones never overlap. ``ref_apex`` is None because the database holds no apex
    speeds; the first completed lap becomes the reference and later laps are compared to it.
    """
    points = sorted(get_braking_points(track_id), key=lambda p: p[0])
    zones: list[dict[str, Any]] = []
    for index, (distance, name, _message) in enumerate(points):
        start = max(0.0, distance - 110.0)
        end = distance + 170.0
        if index + 1 < len(points):
            end = min(end, points[index + 1][0] - 111.0)  # stop before the next zone starts
        if track_length:
            end = min(end, float(track_length) - 1.0)
        if index > 0:
            start = max(start, zones[-1]["end"] + 1.0) if zones else start
        if end - start < 80.0:
            continue
        zones.append(
            {
                "name": name,
                "start": round(start),
                "end": round(end),
                "ref_apex": None,
                "ref_brake": round(distance),
            }
        )
    return zones


def get_track_info(track_id: int) -> dict[str, Any] | None:
    """Get the full metadata dictionary for a track."""
    return TRACK_DATABASE.get(track_id)


def get_all_tracks() -> list[dict[str, Any]]:
    """Get list of all 24 tracks with basic info for dropdown menus."""
    tracks = []
    for tid, t in sorted(TRACK_DATABASE.items(), key=lambda x: x[1]["name"]):
        tracks.append(
            {
                "id": tid,
                "name": t["name"],
                "short_name": t["short_name"],
                "country": t["country"],
                "length_m": t["length_m"],
                "turns": t["turns"],
                "downforce": t["downforce"],
                "tyre_wear": t["tyre_wear"],
                "aliases": t.get("aliases", [t["short_name"]]),
            }
        )
    return tracks


def get_track_setup(track_id: int, weather: str = "dry", session: str = "race") -> dict[str, Any]:
    """Retrieve full 6-category setup with adjustments for weather and session (race or qualifying)."""
    circuit = TRACK_DATABASE.get(resolve_track_id(track_id), TRACK_DATABASE[7])  # default to Silverstone
    setup = circuit["setup"]
    is_wet = "wet" in weather.lower() or "rain" in weather.lower()
    if is_wet:
        from track_analysis import wet_setup  # lazy: track_analysis imports this module

        setup = wet_setup(setup, circuit.get("wet_adjustments"))
    from track_analysis import clamp_setup, normalize_session, qualifying_setup
    from track_profiles import get_profile

    session = normalize_session(session)
    if session == "qualifying":
        base = clamp_setup(setup)
        base["notes"] = setup.get("notes", "")
        setup = qualifying_setup(base, get_profile(circuit["id"]), is_wet)
    return {
        "track_id": circuit["id"],
        "track_name": circuit["name"],
        "country": circuit["country"],
        "characteristics": circuit["characteristics"],
        "downforce_level": circuit["downforce"],
        "weather": "wet" if is_wet else "dry",
        "session": session,
        "setup": setup,
        "wet_adjustments": circuit.get("wet_adjustments", {}) if is_wet else None,
        "engineer_advice": (
            setup.get("notes")
            if session == "qualifying"
            else circuit.get("wet_adjustments", {}).get("notes")
            if is_wet
            else setup.get("notes")
        ),
    }


# ──────────────────────────────────────────────
# AI Setup Advisor (Prompt Construction & Generation)
# ──────────────────────────────────────────────

SETUP_SYSTEM_PROMPT = """You are a senior Formula 1 Chief Race Engineer.
Provide a concise, high-performance car setup breakdown for the driver tailored to the specified circuit and conditions.
Cover the 6 official F1 2021 setup parameters:
1. Aerodynamics (Front Wing / Rear Wing: 1-11)
2. Transmission (On-Throttle Differential / Off-Throttle Differential: 50%-100%)
3. Suspension Geometry (Camber & Toe angles)
4. Suspension (Front & Rear Suspension, Anti-Roll Bars, Ride Heights: 1-11)
5. Brakes (Brake Pressure 80-100%, Brake Bias 50-70%)
6. Tyre Pressures (PSI)
Explain why this setup works for this specific track's layout and weather in 2-3 concise sentences.
Keep your output structured, technical, and realistic for F1 2021."""


def build_setup_advisor_prompt(
    track_id: int, weather: str = "dry", handling_issue: str = "balanced"
) -> tuple[str, str]:
    """Build system and user prompts for DeepSeek / AI model to advise on setup."""
    circuit = TRACK_DATABASE.get(resolve_track_id(track_id), TRACK_DATABASE[7])
    from track_analysis import model_analysis  # lazy: track_analysis imports this module

    tyre_deg = model_analysis(circuit["id"], weather)["tyre_degradation"]
    user_prompt = (
        f"Circuit: {circuit['name']} ({circuit['country']})\n"
        f"Length: {circuit['length_m']}m, {circuit['turns']} Turns\n"
        f"Track Profile: {circuit['characteristics']}\n"
        f"Baseline Downforce Level: {circuit['downforce']}\n"
        f"Tyre Degradation ({weather}): {tyre_deg}\n"
        f"Weather Conditions: {weather}\n"
        f"Driver Balance Preference: {handling_issue}\n\n"
        f"Please provide the recommended 6-category F1 2021 car setup values and tactical setup notes."
    )
    return SETUP_SYSTEM_PROMPT, user_prompt


def generate_ai_setup_recommendation(
    track_id: int, weather: str = "dry", handling_issue: str = "balanced", ai_client=None
) -> dict[str, Any]:
    """Generate setup recommendation using AI model or fallback engineered baseline."""
    track_info = get_track_setup(track_id, weather)
    if ai_client is not None:
        try:
            sys_prompt, user_prompt = build_setup_advisor_prompt(track_id, weather, handling_issue)
            ai_notes = ai_client.chat(sys_prompt, user_prompt, max_tokens=1500)
            if ai_notes and len(ai_notes.strip()) > 20:
                track_info["ai_consultation"] = ai_notes.strip()
                return track_info
        except Exception as e:
            track_info["ai_consultation_error"] = str(e)

    # Fallback to high quality engineered commentary
    notes = track_info["engineer_advice"]
    track_info["ai_consultation"] = (
        f"Engineer Setup Briefing for {track_info['track_name']} ({weather.upper()}):\n"
        f"{notes}\n"
        f"Aerodynamics set to {track_info['setup']['aerodynamics']['front_wing']}/{track_info['setup']['aerodynamics']['rear_wing']} "
        f"to match the {track_info['downforce_level'].lower()} downforce demand. "
        f"Differential at {track_info['setup']['transmission']['diff_on_throttle']}% on-throttle for progressive traction."
    )
    return track_info
