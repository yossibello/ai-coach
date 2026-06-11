"""
Cycling capability predictor.

Given FTP, weight, CTL, power curve and longest-ride history, estimates
what events the athlete can complete and at what tier (gold/silver/bronze).

Physics reference:
  P = CdA·ρ·v³/2 + Crr·m·g·v + m·g·sin(θ)·v
  CdA=0.32 m² (hoods), ρ=1.2 kg/m³, Crr=0.004, η_dt=0.97
"""
from __future__ import annotations
import math
from typing import Optional

# ── Cycling physics ────────────────────────────────────────────────────────────

CDA    = 0.32   # drag area m² (hoods position, typical road bike)
RHO    = 1.2    # air density kg/m³
CRR    = 0.004  # rolling resistance (road surface)
G      = 9.81   # gravity m/s²
ETA_DT = 0.97   # drivetrain efficiency

def flat_speed_kmh(power_w: float, mass_kg: float) -> float:
    """Return sustainable speed on flat road at given power (binary search)."""
    lo, hi = 5.0, 70.0
    for _ in range(60):
        mid = (lo + hi) / 2
        v = mid / 3.6
        p_needed = (0.5 * CDA * RHO * v**3 + CRR * mass_kg * G * v) / ETA_DT
        if p_needed < power_w:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 1)

def climb_speed_kmh(wkg: float, gradient: float = 0.07) -> float:
    """Return climbing speed (km/h) at W/kg and given gradient (fraction)."""
    # Simplified: power ≈ mass·g·gradient·v / eta (aerodrag small on climbs)
    # v = wkg / (g · gradient / eta)
    v_ms = wkg * ETA_DT / (G * gradient)
    return round(v_ms * 3.6, 1)

def max_duration_h(ctl: float) -> float:
    """Empirical max sustainable ride duration from CTL (Bannister model proxy)."""
    # Literature: CTL ≈ daily chronic load; max duration scales roughly as:
    # Seiler (2010): well-trained athletes can sustain 3-5× weekly-avg duration
    # Approximation tuned to known athlete profiles
    if ctl < 20:  return 3.0
    if ctl < 35:  return 4.5
    if ctl < 50:  return 6.5
    if ctl < 65:  return 9.0
    if ctl < 80:  return 12.0
    if ctl < 95:  return 16.0
    return 20.0

def estimate_event_time_h(
    distance_km: float, elevation_m: float,
    sustained_wkg: float, mass_kg: float
) -> float:
    """Rough event completion time using flat + climbing split."""
    # Gradient correction: elevation / (distance * 1000) ≈ avg gradient fraction
    avg_grad = elevation_m / (distance_km * 1000) if distance_km > 0 else 0
    # Flat equivalent distance (remove vertical component from flat speed calc)
    flat_km   = distance_km * 0.7  # assume 70% rolling/descent
    climb_km  = distance_km * 0.3  # 30% climbing
    flat_spd  = flat_speed_kmh(sustained_wkg * mass_kg, mass_kg)
    climb_spd = climb_speed_kmh(sustained_wkg, max(avg_grad, 0.05))
    t_flat    = flat_km  / flat_spd  if flat_spd  > 0 else 999
    t_climb   = climb_km / climb_spd if climb_spd > 0 else 999
    return round(t_flat + t_climb, 1)


# ── Events database ────────────────────────────────────────────────────────────

EVENTS: list[dict] = [
    # ── Group rides ──────────────────────────────────────────────────────────
    {
        "id": "coffee_ride",
        "name": "Coffee Shop Ride",
        "category": "group_ride",
        "icon": "☕",
        "description": "Casual weekend group ride, chat pace, cafe stop included.",
        "distance_km": 60, "elevation_m": 600,
        "tiers": [
            # gold: lead the group  silver: comfortable  bronze: can complete
            {"label": "gold",   "emoji": "🥇", "name": "Pull the bunch",       "min_flat_kmh": 34, "min_ctl": 40},
            {"label": "silver", "emoji": "🥈", "name": "Comfortable in bunch", "min_flat_kmh": 29, "min_ctl": 25},
            {"label": "bronze", "emoji": "🥉", "name": "Can complete",         "min_flat_kmh": 24, "min_ctl": 10},
        ],
        "type": "flat_speed",
    },
    {
        "id": "fast_group",
        "name": "Fast Club Ride",
        "category": "group_ride",
        "icon": "🚴",
        "description": "Hammerfest-style club ride. No-drop but no mercy either.",
        "distance_km": 100, "elevation_m": 800,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Sets the pace",     "min_flat_kmh": 42, "min_ctl": 70},
            {"label": "silver", "emoji": "🥈", "name": "Hangs on all day",  "min_flat_kmh": 37, "min_ctl": 50},
            {"label": "bronze", "emoji": "🥉", "name": "Survives the ride", "min_flat_kmh": 32, "min_ctl": 35},
        ],
        "type": "flat_speed",
    },
    {
        "id": "racing_peloton",
        "name": "Racing Peloton",
        "category": "group_ride",
        "icon": "⚡",
        "description": "Criterium or road race peloton. Full gas, no excuses.",
        "distance_km": 120, "elevation_m": 600,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Contends for win",  "min_flat_kmh": 46, "min_ctl": 90},
            {"label": "silver", "emoji": "🥈", "name": "Survives the race", "min_flat_kmh": 42, "min_ctl": 70},
            {"label": "bronze", "emoji": "🥉", "name": "Makes the start",   "min_flat_kmh": 38, "min_ctl": 55},
        ],
        "type": "flat_speed",
    },

    # ── Gran Fondos ───────────────────────────────────────────────────────────
    {
        "id": "la_marmotte",
        "name": "La Marmotte",
        "category": "gran_fondo",
        "icon": "🏔️",
        "description": "174km, 5,000m. Galibier, Croix-de-Fer, Alpe d'Huez. The classic.",
        "distance_km": 174, "elevation_m": 5000,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Gold Marmotte (<8h)",   "min_wkg": 3.8, "min_ctl": 70, "max_time_h": 8.0},
            {"label": "silver", "emoji": "🥈", "name": "Silver (<10h)",         "min_wkg": 3.1, "min_ctl": 55, "max_time_h": 10.0},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher (<13h)",       "min_wkg": 2.5, "min_ctl": 40, "max_time_h": 13.0},
        ],
        "type": "climb",
    },
    {
        "id": "oetztaler",
        "name": "Ötztaler Radmarathon",
        "category": "gran_fondo",
        "icon": "🦅",
        "description": "238km, 5,500m. Austria's hardest sportive. Four major Alpine passes.",
        "distance_km": 238, "elevation_m": 5500,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Sub 9h",     "min_wkg": 4.2, "min_ctl": 80, "max_time_h": 9.0},
            {"label": "silver", "emoji": "🥈", "name": "Sub 12h",    "min_wkg": 3.3, "min_ctl": 65, "max_time_h": 12.0},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher",   "min_wkg": 2.7, "min_ctl": 50, "max_time_h": 18.0},
        ],
        "type": "climb",
    },
    {
        "id": "etape_du_tour",
        "name": "L'Étape du Tour",
        "category": "gran_fondo",
        "icon": "🇫🇷",
        "description": "Ride a real Tour de France stage before the pros. Varies yearly (~160km, 4,000m).",
        "distance_km": 165, "elevation_m": 4200,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Top 10% (<7h30)",  "min_wkg": 3.9, "min_ctl": 75, "max_time_h": 7.5},
            {"label": "silver", "emoji": "🥈", "name": "Top 30% (<9h)",    "min_wkg": 3.2, "min_ctl": 60, "max_time_h": 9.0},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher (<11h)",  "min_wkg": 2.6, "min_ctl": 45, "max_time_h": 11.0},
        ],
        "type": "climb",
    },

    # ── Gravel races ─────────────────────────────────────────────────────────
    {
        "id": "unbound_200",
        "name": "Unbound Gravel 200",
        "category": "gravel",
        "icon": "🪨",
        "description": "320km of Kansas flint gravel. The world's hardest gravel race. One day.",
        "distance_km": 320, "elevation_m": 4000,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Top finisher (<14h)", "min_wkg": 3.8, "min_ctl": 90,  "max_duration_h": 14},
            {"label": "silver", "emoji": "🥈", "name": "Mid-pack (<18h)",     "min_wkg": 3.0, "min_ctl": 75,  "max_duration_h": 18},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher (<24h)",     "min_wkg": 2.4, "min_ctl": 60,  "max_duration_h": 24},
        ],
        "type": "ultra",
    },
    {
        "id": "belgian_waffle",
        "name": "Belgian Waffle Ride",
        "category": "gravel",
        "icon": "🧇",
        "description": "200km, punchy climbs, cobbles, gravel. California suffering.",
        "distance_km": 200, "elevation_m": 3500,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Top 15% (<9h)",   "min_wkg": 3.6, "min_ctl": 75, "max_time_h": 9.0},
            {"label": "silver", "emoji": "🥈", "name": "Top 50% (<12h)",  "min_wkg": 2.9, "min_ctl": 58, "max_time_h": 12.0},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher (<16h)", "min_wkg": 2.3, "min_ctl": 45, "max_time_h": 16.0},
        ],
        "type": "climb",
    },

    # ── Swedish classics ─────────────────────────────────────────────────────
    {
        "id": "vatternrundan",
        "name": "Vätternrundan",
        "category": "gran_fondo",
        "icon": "🇸🇪",
        "description": "315km around Lake Vättern. World's largest cycling event. Flat but relentless — the distance breaks you, not the hills.",
        "distance_km": 315, "elevation_m": 1800,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Sub 10h (~31 km/h)", "min_flat_kmh": 32, "min_ctl": 72, "max_time_h": 10.0},
            {"label": "silver", "emoji": "🥈", "name": "Sub 13h (~24 km/h)", "min_flat_kmh": 25, "min_ctl": 52, "max_time_h": 13.0},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher <18h",      "min_flat_kmh": 18, "min_ctl": 35, "max_time_h": 18.0},
        ],
        "type": "flat_speed",
    },
    {
        "id": "halvvattern",
        "name": "Halvvättern",
        "category": "gran_fondo",
        "icon": "🏅",
        "description": "150km around southern Lake Vättern. The classic gateway to Vätternrundan — same roads, half the suffering.",
        "distance_km": 150, "elevation_m": 900,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Sub 5h (~30 km/h)", "min_flat_kmh": 31, "min_ctl": 58, "max_time_h": 5.0},
            {"label": "silver", "emoji": "🥈", "name": "Sub 7h (~21 km/h)", "min_flat_kmh": 22, "min_ctl": 40, "max_time_h": 7.0},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher <10h",     "min_flat_kmh": 15, "min_ctl": 25, "max_time_h": 10.0},
        ],
        "type": "flat_speed",
    },
    {
        "id": "cykelvasan",
        "name": "Cykelvasan",
        "category": "gravel",
        "icon": "🌲",
        "description": "90km gravel through Dalarna forests — same trail as the legendary Vasaloppet ski race. Sälen to Mora.",
        "distance_km": 90, "elevation_m": 560,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Sub 3h",          "min_wkg": 3.9, "min_ctl": 68, "max_time_h": 3.0},
            {"label": "silver", "emoji": "🥈", "name": "Sub 4h",          "min_wkg": 3.0, "min_ctl": 50, "max_time_h": 4.0},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher <5h30",  "min_wkg": 2.3, "min_ctl": 36, "max_time_h": 5.5},
        ],
        "type": "climb",
    },
    {
        "id": "tjejvattern",
        "name": "Tjejvättern",
        "category": "gran_fondo",
        "icon": "💜",
        "description": "100km around Lake Vättern. Scandinavia's biggest women's cycling event — open to all, celebrated by all.",
        "distance_km": 100, "elevation_m": 600,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Sub 3h15 (~31 km/h)", "min_flat_kmh": 32, "min_ctl": 52, "max_time_h": 3.25},
            {"label": "silver", "emoji": "🥈", "name": "Sub 4h30 (~22 km/h)", "min_flat_kmh": 23, "min_ctl": 35, "max_time_h": 4.5},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher <6h",        "min_flat_kmh": 17, "min_ctl": 22, "max_time_h": 6.0},
        ],
        "type": "flat_speed",
    },

    # ── Ultra distance ────────────────────────────────────────────────────────
    {
        "id": "paris_brest_paris",
        "name": "Paris-Brest-Paris",
        "category": "ultra",
        "icon": "🌙",
        "description": "1,200km, 90h cutoff. Sleep-deprived randonnée across France and back.",
        "distance_km": 1200, "elevation_m": 11000,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "Sub 60h",       "min_ctl": 80, "max_duration_h": 60},
            {"label": "silver", "emoji": "🥈", "name": "Sub 80h",       "min_ctl": 60, "max_duration_h": 80},
            {"label": "bronze", "emoji": "🥉", "name": "Finisher <90h", "min_ctl": 45, "max_duration_h": 90},
        ],
        "type": "ultra_endurance",
    },
    {
        "id": "transcontinental",
        "name": "Ultra 24h TT",
        "category": "ultra",
        "icon": "🌍",
        "description": "24-hour solo time trial. How far can you go without stopping?",
        "distance_km": None, "elevation_m": None,
        "tiers": [
            {"label": "gold",   "emoji": "🥇", "name": "600km+ (elite)",   "min_ctl": 90,  "min_distance_km": 600},
            {"label": "silver", "emoji": "🥈", "name": "450km+ (serious)", "min_ctl": 70,  "min_distance_km": 450},
            {"label": "bronze", "emoji": "🥉", "name": "300km+ (finisher)","min_ctl": 50,  "min_distance_km": 300},
        ],
        "type": "ultra_24h",
    },
]


# ── Capability evaluation ──────────────────────────────────────────────────────

def evaluate(
    ftp_w: float,
    weight_kg: float,
    ctl: float,
    *,
    pc_5min_wkg: Optional[float] = None,
    pc_1min_wkg: Optional[float] = None,
) -> list[dict]:
    """Return capability assessment for all events."""
    if weight_kg <= 0:
        weight_kg = 70.0

    wkg = ftp_w / weight_kg
    # Sustained power fraction depends on duration; use 75% FTP for flat group rides
    flat_power_sustained = ftp_w * 0.75
    flat_spd = flat_speed_kmh(flat_power_sustained, weight_kg)
    max_dur  = max_duration_h(ctl)

    results = []
    for ev in EVENTS:
        tier_achieved = None
        est_speed     = None
        est_time_h    = None

        # For flat_speed events, show the GOLD tier required speed (what the event demands)
        if ev["type"] == "flat_speed":
            est_speed = ev["tiers"][0]["min_flat_kmh"]  # gold threshold = event benchmark speed

        for tier in ev["tiers"]:
            achieved = False

            if ev["type"] == "flat_speed":
                achieved = (flat_spd >= tier["min_flat_kmh"]
                            and ctl >= tier["min_ctl"])

            elif ev["type"] == "climb":
                # Use 95% FTP for sustained climbing (typical long climb intensity)
                sustained_wkg = wkg * 0.95
                est_time_h = estimate_event_time_h(
                    ev["distance_km"], ev["elevation_m"], sustained_wkg, weight_kg
                )
                achieved = (wkg >= tier["min_wkg"]
                            and ctl >= tier["min_ctl"]
                            and est_time_h <= tier["max_time_h"]
                            and max_dur >= ev["distance_km"] / max(est_time_h, 0.1) * 0.1)

            elif ev["type"] == "ultra":
                sustained_wkg = wkg * 0.70  # ultra = 70% FTP sustained
                est_time_h = estimate_event_time_h(
                    ev["distance_km"], ev["elevation_m"], sustained_wkg, weight_kg
                )
                achieved = (wkg >= tier["min_wkg"]
                            and ctl >= tier["min_ctl"]
                            and max_dur >= tier["max_duration_h"] * 0.7)

            elif ev["type"] == "ultra_endurance":
                achieved = (ctl >= tier["min_ctl"]
                            and max_dur >= 18)

            elif ev["type"] == "ultra_24h":
                # Estimate 24h distance: ~75% of max_dur as sustainable fraction
                daily_speed = flat_speed_kmh(ftp_w * 0.65, weight_kg)
                est_km_24h  = daily_speed * min(max_dur, 22)  # ride 22 of 24h
                achieved = (ctl >= tier["min_ctl"]
                            and est_km_24h >= tier["min_distance_km"])

            if achieved:
                tier_achieved = tier
                break  # tiers are ordered gold→silver→bronze; stop at first match

        # Next tier to unlock (what to aim for)
        current_idx  = next((i for i, t in enumerate(ev["tiers"]) if t == tier_achieved), None)
        next_tier    = ev["tiers"][current_idx - 1] if (current_idx is not None and current_idx > 0) else (
            ev["tiers"][0] if tier_achieved is None else None
        )

        results.append({
            "id":                ev["id"],
            "name":              ev["name"],
            "category":          ev["category"],
            "icon":              ev["icon"],
            "description":       ev["description"],
            "distance_km":       ev.get("distance_km"),
            "elevation_m":       ev.get("elevation_m"),
            "tier":              tier_achieved["label"]   if tier_achieved else None,
            "tier_emoji":        tier_achieved["emoji"]   if tier_achieved else "—",
            "tier_name":         tier_achieved["name"]    if tier_achieved else "Not ready yet",
            "next_tier":         next_tier["label"]       if next_tier else None,
            "next_tier_name":    next_tier["name"]        if next_tier else None,
            "event_speed_kmh":   est_speed,      # what the event demands (gold benchmark)
            "athlete_speed_kmh": flat_spd,       # what the athlete can sustain
            "est_time_h":        est_time_h,
            "athlete_wkg":       round(wkg, 2),
            "athlete_flat_kmh":  flat_spd,
            "athlete_max_dur_h": max_dur,
        })

    return results
