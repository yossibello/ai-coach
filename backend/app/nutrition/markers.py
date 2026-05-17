"""
Blood-marker reference ranges and athlete-optimal targets.

For each marker:
  - aliases: substrings the PDF parser will match (case-insensitive)
  - units: canonical unit; values in other units are converted on ingest
  - ref_low / ref_high: standard clinical reference range (adult)
  - athlete_optimal_low / athlete_optimal_high: tighter range for endurance athletes
    (where research supports it). For markers without sport-specific evidence,
    these equal the clinical range.
  - critical_low / critical_high: trigger urgent "see physician" warnings
  - sex_specific: dict of {"male": (low, high), "female": (low, high)} if applicable

Status logic (status_for_value):
  value < critical_low                → "critical_low"
  value > critical_high               → "critical_high"
  value < ref_low                     → "low"
  value > ref_high                    → "high"
  athlete_optimal_low ≤ v ≤ athlete_optimal_high → "optimal"
  otherwise (in clinical range, outside athlete optimal) → "suboptimal"
"""
from __future__ import annotations

from typing import Optional


# Each marker dict shape — keep it simple JSON-compatible
MARKERS: dict[str, dict] = {
    # ─── Iron panel ──────────────────────────────────────────────────────────
    "ferritin": {
        "label": "Ferritin",
        "category": "iron",
        "aliases": ["ferritin", "s-ferritin", "p-ferritin"],
        "unit": "ng/mL",
        "ref_low": 30, "ref_high": 400,
        "athlete_optimal_low": 50, "athlete_optimal_high": 200,
        "critical_low": 12, "critical_high": 1000,
        "sex_specific": {"female": (15, 200), "male": (30, 400)},
        "performance_note": "Below 35 ng/mL = stage-2 iron deficiency, well-documented "
                            "to impair endurance even without anemia (Sim 2019).",
    },
    "serum_iron": {
        "label": "Serum Iron",
        "category": "iron",
        "aliases": ["serum iron", "iron, serum", "iron level", "s-järn", "p-järn", "järn"],
        "unit": "µg/dL",
        "ref_low": 60, "ref_high": 170,
        "athlete_optimal_low": 80, "athlete_optimal_high": 160,
        "critical_low": 30, "critical_high": 250,
    },
    "transferrin_saturation": {
        "label": "Transferrin Saturation",
        "category": "iron",
        "aliases": ["transferrin saturation", "tsat", "tf sat", "% saturation", "transferrinmättnad", "järnmättnad"],
        "unit": "%",
        "ref_low": 20, "ref_high": 50,
        "athlete_optimal_low": 25, "athlete_optimal_high": 45,
        "critical_low": 10, "critical_high": 60,
    },
    "tibc": {
        "label": "Total Iron Binding Capacity",
        "category": "iron",
        "aliases": ["tibc", "total iron binding", "total järnbindningskapacitet", "järnbindningskapacitet"],
        "unit": "µg/dL",
        "ref_low": 240, "ref_high": 450,
        "athlete_optimal_low": 240, "athlete_optimal_high": 450,
        "critical_low": 150, "critical_high": 600,
    },
    "hemoglobin": {
        "label": "Hemoglobin",
        "category": "iron",
        "aliases": ["hemoglobin", "haemoglobin", "hgb", "hb", "b-hb", "b-hemoglobin"],
        "unit": "g/dL",
        "ref_low": 12.0, "ref_high": 17.5,
        "athlete_optimal_low": 13.5, "athlete_optimal_high": 17.0,
        "critical_low": 10.0, "critical_high": 19.0,
        "sex_specific": {"female": (12.0, 15.5), "male": (13.5, 17.5)},
    },
    "hematocrit": {
        "label": "Hematocrit",
        "category": "iron",
        "aliases": ["hematocrit", "haematocrit", "hct", "hematokrit", "evf", "erytrocyt-volym-fraktion"],
        "unit": "%",
        "ref_low": 36, "ref_high": 50,
        "athlete_optimal_low": 40, "athlete_optimal_high": 50,
        "critical_low": 30, "critical_high": 55,
        "sex_specific": {"female": (36, 46), "male": (41, 50)},
    },

    # ─── Fat-soluble vitamins ────────────────────────────────────────────────
    "vitamin_d": {
        "label": "Vitamin D (25-OH)",
        "category": "vitamins",
        "aliases": ["25-oh", "25(oh)d", "vitamin d", "vit d", "calcidiol", "25 hydroxy", "kalcidiol", "d-vitamin", "25-oh-vitamin d", "s-25-oh vitamin d"],
        "unit": "ng/mL",
        "ref_low": 30, "ref_high": 100,
        "athlete_optimal_low": 30, "athlete_optimal_high": 80,
        "critical_low": 20, "critical_high": 150,
        "performance_note": "<30 ng/mL is associated with impaired bone health, "
                            "muscle function, and immunity (Owens 2018).",
    },

    # ─── B-vitamins ──────────────────────────────────────────────────────────
    "vitamin_b12": {
        "label": "Vitamin B12",
        "category": "vitamins",
        "aliases": ["b12", "cobalamin", "vitamin b-12", "kobalamin", "s-kobalamin"],
        "unit": "pg/mL",
        "ref_low": 200, "ref_high": 900,
        "athlete_optimal_low": 300, "athlete_optimal_high": 900,
        "critical_low": 150, "critical_high": 2000,
        "performance_note": "Vegan/vegetarian athletes are at high risk; "
                            "deficiency causes macrocytic anemia and fatigue.",
    },
    "folate": {
        "label": "Folate (B9)",
        "category": "vitamins",
        "aliases": ["folate", "folic acid", "b9", "folat", "folsyra", "s-folat"],
        "unit": "ng/mL",
        "ref_low": 3, "ref_high": 17,
        "athlete_optimal_low": 5, "athlete_optimal_high": 17,
        "critical_low": 2, "critical_high": 25,
    },
    "vitamin_b6": {
        "label": "Vitamin B6",
        "category": "vitamins",
        "aliases": ["b6", "pyridoxine", "p5p", "vitamin b-6", "pyridoxin"],
        "unit": "ng/mL",
        "ref_low": 5, "ref_high": 50,
        "athlete_optimal_low": 8, "athlete_optimal_high": 50,
        "critical_low": 3, "critical_high": 200,
    },

    # ─── Minerals ────────────────────────────────────────────────────────────
    "magnesium": {
        "label": "Magnesium (serum)",
        "category": "minerals",
        "aliases": ["magnesium", "mg, serum", "magnesium serum", "s-magnesium", "p-magnesium"],
        "unit": "mg/dL",
        "ref_low": 1.7, "ref_high": 2.4,
        "athlete_optimal_low": 1.9, "athlete_optimal_high": 2.4,
        "critical_low": 1.4, "critical_high": 2.8,
        "performance_note": "Sweat losses are significant; serum is a poor proxy "
                            "(only 1% of body Mg is in serum) but low values are "
                            "still meaningful.",
    },
    "magnesium_rbc": {
        "label": "Magnesium RBC",
        "category": "minerals",
        "aliases": ["magnesium rbc", "rbc magnesium", "intracellular magnesium"],
        "unit": "mg/dL",
        "ref_low": 4.0, "ref_high": 6.4,
        "athlete_optimal_low": 5.0, "athlete_optimal_high": 6.4,
        "critical_low": 3.5, "critical_high": 7.0,
    },
    "zinc": {
        "label": "Zinc",
        "category": "minerals",
        "aliases": ["zinc", "zn", "zink", "s-zink"],
        "unit": "µg/dL",
        "ref_low": 70, "ref_high": 120,
        "athlete_optimal_low": 80, "athlete_optimal_high": 120,
        "critical_low": 50, "critical_high": 200,
    },
    "calcium": {
        "label": "Calcium",
        "category": "minerals",
        "aliases": ["calcium, total", "calcium total", "ca, total", "calcium", "kalcium", "s-kalcium", "p-kalcium"],
        "unit": "mg/dL",
        "ref_low": 8.6, "ref_high": 10.3,
        "athlete_optimal_low": 9.0, "athlete_optimal_high": 10.3,
        "critical_low": 7.5, "critical_high": 11.5,
    },
    "calcium_ionized": {
        "label": "Calcium (ionized)",
        "category": "minerals",
        "aliases": ["ionized calcium", "ionised calcium", "calcium, ionized", "joniserat calcium", "s-joniserat calcium", "ca2+", "ca++"],
        "unit": "mmol/L",
        "ref_low": 1.12, "ref_high": 1.32,
        "athlete_optimal_low": 1.15, "athlete_optimal_high": 1.30,
        "critical_low": 0.80, "critical_high": 1.60,
    },
    "sodium": {
        "label": "Sodium",
        "category": "electrolytes",
        "aliases": ["sodium", "na", "na+", "natrium", "s-natrium", "p-natrium"],
        "unit": "mmol/L",
        "ref_low": 135, "ref_high": 145,
        "athlete_optimal_low": 137, "athlete_optimal_high": 143,
        "critical_low": 130, "critical_high": 150,
    },
    "potassium": {
        "label": "Potassium",
        "category": "electrolytes",
        "aliases": ["potassium", "k+", "k,", "kalium", "s-kalium", "p-kalium"],
        "unit": "mmol/L",
        "ref_low": 3.5, "ref_high": 5.1,
        "athlete_optimal_low": 4.0, "athlete_optimal_high": 5.0,
        "critical_low": 3.0, "critical_high": 5.5,
    },

    # ─── Thyroid ─────────────────────────────────────────────────────────────
    "tsh": {
        "label": "TSH",
        "category": "thyroid",
        "aliases": ["tsh", "thyroid stimulating", "thyrotropin", "tyreotropin", "s-tsh"],
        "unit": "mIU/L",
        "ref_low": 0.4, "ref_high": 4.5,
        "athlete_optimal_low": 0.8, "athlete_optimal_high": 2.5,
        "critical_low": 0.1, "critical_high": 10.0,
        "performance_note": "TSH > 2.5 with low T3/T4 may indicate subclinical "
                            "hypothyroidism — affects VO2max and recovery.",
    },
    "free_t3": {
        "label": "Free T3",
        "category": "thyroid",
        "aliases": ["free t3", "ft3", "triiodothyronine, free", "fritt t3", "fritt trijodtyronin"],
        "unit": "pg/mL",
        "ref_low": 2.0, "ref_high": 4.4,
        "athlete_optimal_low": 2.8, "athlete_optimal_high": 4.4,
        "critical_low": 1.5, "critical_high": 6.0,
    },
    "free_t4": {
        "label": "Free T4",
        "category": "thyroid",
        "aliases": ["free t4", "ft4", "thyroxine, free", "fritt t4", "fritt tyroxin", "s-fritt t4", "tyroxin"],
        "unit": "ng/dL",
        "ref_low": 0.8, "ref_high": 1.8,
        "athlete_optimal_low": 1.0, "athlete_optimal_high": 1.6,
        "critical_low": 0.5, "critical_high": 3.0,
    },

    # ─── Hormones / RED-S markers ────────────────────────────────────────────
    "testosterone_total": {
        "label": "Testosterone (total)",
        "category": "hormones",
        "aliases": ["testosterone, total", "total testosterone", "testosteron", "s-testosteron"],
        "unit": "ng/dL",
        "ref_low": 264, "ref_high": 916,
        "athlete_optimal_low": 400, "athlete_optimal_high": 900,
        "critical_low": 200, "critical_high": 1200,
        "sex_specific": {"female": (15, 70), "male": (264, 916)},
        "performance_note": "Chronically low T in male endurance athletes is a key "
                            "RED-S (Relative Energy Deficiency in Sport) marker.",
    },
    "testosterone_free": {
        "label": "Testosterone (free)",
        "category": "hormones",
        "aliases": ["free testosterone", "testosterone, free", "fritt testosteron"],
        "unit": "pg/mL",
        "ref_low": 50, "ref_high": 210,
        "athlete_optimal_low": 80, "athlete_optimal_high": 210,
        "critical_low": 30, "critical_high": 300,
    },
    "cortisol": {
        "label": "Cortisol (AM)",
        "category": "hormones",
        "aliases": ["cortisol", "cortisol am", "morning cortisol", "kortisol", "s-kortisol"],
        "unit": "µg/dL",
        "ref_low": 6, "ref_high": 23,
        "athlete_optimal_low": 8, "athlete_optimal_high": 18,
        "critical_low": 3, "critical_high": 30,
        "performance_note": "Persistently elevated AM cortisol with low T can "
                            "indicate overtraining syndrome.",
    },
    "shbg": {
        "label": "SHBG",
        "category": "hormones",
        "aliases": ["shbg", "sex hormone binding globulin"],
        "unit": "nmol/L",
        "ref_low": 10, "ref_high": 80,
        "athlete_optimal_low": 15, "athlete_optimal_high": 60,
        "critical_low": 5, "critical_high": 120,
    },

    # ─── Muscle / liver / kidney ─────────────────────────────────────────────
    "ck": {
        "label": "Creatine Kinase (CK)",
        "category": "muscle",
        "aliases": ["creatine kinase", "ck", "cpk", "kreatinkinas", "s-kreatinkinas"],
        "unit": "U/L",
        "ref_low": 30, "ref_high": 200,
        "athlete_optimal_low": 30, "athlete_optimal_high": 400,   # athletes run higher baseline
        "critical_low": 0, "critical_high": 1000,
        "performance_note": "Athletes typically have baseline CK 200-400 U/L. "
                            ">1000 U/L = significant muscle damage; investigate.",
    },
    "ldh": {
        "label": "LDH",
        "category": "muscle",
        "aliases": ["ldh", "lactate dehydrogenase", "laktatdehydrogenas", "s-ld ", "p-ld "],
        "unit": "U/L",
        "ref_low": 122, "ref_high": 222,
        "athlete_optimal_low": 122, "athlete_optimal_high": 250,
        "critical_low": 80, "critical_high": 400,
    },
    "ast": {
        "label": "AST",
        "category": "liver",
        "aliases": ["ast", "sgot", "aspartate", "asat", "s-asat", "p-asat"],
        "unit": "U/L",
        "ref_low": 10, "ref_high": 40,
        "athlete_optimal_low": 10, "athlete_optimal_high": 50,
        "critical_low": 5, "critical_high": 200,
    },
    "alt": {
        "label": "ALT",
        "category": "liver",
        "aliases": ["alt", "sgpt", "alanine", "alat", "s-alat", "p-alat"],
        "unit": "U/L",
        "ref_low": 7, "ref_high": 56,
        "athlete_optimal_low": 7, "athlete_optimal_high": 56,
        "critical_low": 5, "critical_high": 200,
    },
    "creatinine": {
        "label": "Creatinine",
        "category": "kidney",
        "aliases": ["creatinine", "kreatinin", "s-kreatinin", "p-kreatinin"],
        "unit": "mg/dL",
        "ref_low": 0.6, "ref_high": 1.3,
        "athlete_optimal_low": 0.7, "athlete_optimal_high": 1.4,   # higher muscle mass → higher
        "critical_low": 0.4, "critical_high": 2.0,
    },
    "egfr": {
        "label": "eGFR",
        "category": "kidney",
        "aliases": ["egfr", "estimated gfr"],
        "unit": "mL/min/1.73m²",
        "ref_low": 90, "ref_high": 200,
        "athlete_optimal_low": 90, "athlete_optimal_high": 200,
        "critical_low": 60, "critical_high": 250,
    },
    "urea": {
        "label": "Urea (BUN)",
        "category": "kidney",
        "aliases": ["urea", "bun", "blood urea nitrogen", "karbamid", "s-karbamid", "p-karbamid"],
        "unit": "mg/dL",
        "ref_low": 7, "ref_high": 20,
        "athlete_optimal_low": 7, "athlete_optimal_high": 22,
        "critical_low": 4, "critical_high": 40,
    },

    # ─── Glucose / metabolic ─────────────────────────────────────────────────
    "glucose_fasting": {
        "label": "Glucose (fasting)",
        "category": "metabolic",
        "aliases": ["glucose, fasting", "fasting glucose", "glucose", "glukos", "p-glukos", "fp-glukos"],
        "unit": "mg/dL",
        "ref_low": 70, "ref_high": 99,
        "athlete_optimal_low": 75, "athlete_optimal_high": 95,
        "critical_low": 50, "critical_high": 126,
    },
    "hba1c": {
        "label": "HbA1c",
        "category": "metabolic",
        "aliases": ["hba1c", "a1c", "glycated hemoglobin", "glycohemoglobin", "b-hba1c"],
        "unit": "%",
        "ref_low": 4.0, "ref_high": 5.6,
        "athlete_optimal_low": 4.5, "athlete_optimal_high": 5.4,
        "critical_low": 3.5, "critical_high": 6.5,
    },

    # ─── Lipids ──────────────────────────────────────────────────────────────
    "total_cholesterol": {
        "label": "Total Cholesterol",
        "category": "lipids",
        "aliases": ["total cholesterol", "cholesterol, total", "kolesterol", "kolesterol total", "s-kolesterol"],
        "unit": "mg/dL",
        "ref_low": 100, "ref_high": 200,
        "athlete_optimal_low": 140, "athlete_optimal_high": 200,
        "critical_low": 80, "critical_high": 280,
    },
    "ldl": {
        "label": "LDL Cholesterol",
        "category": "lipids",
        "aliases": ["ldl", "ldl cholesterol", "ldl-kolesterol"],
        "unit": "mg/dL",
        "ref_low": 0, "ref_high": 100,
        "athlete_optimal_low": 0, "athlete_optimal_high": 100,
        "critical_low": 0, "critical_high": 190,
    },
    "hdl": {
        "label": "HDL Cholesterol",
        "category": "lipids",
        "aliases": ["hdl", "hdl cholesterol", "hdl-kolesterol"],
        "unit": "mg/dL",
        "ref_low": 40, "ref_high": 100,
        "athlete_optimal_low": 50, "athlete_optimal_high": 100,
        "critical_low": 25, "critical_high": 150,
    },
    "triglycerides": {
        "label": "Triglycerides",
        "category": "lipids",
        "aliases": ["triglycerides", "tg, ", "triglycerider", "s-triglycerider"],
        "unit": "mg/dL",
        "ref_low": 0, "ref_high": 150,
        "athlete_optimal_low": 0, "athlete_optimal_high": 100,
        "critical_low": 0, "critical_high": 500,
    },

    # ─── Inflammation ────────────────────────────────────────────────────────
    "crp": {
        "label": "CRP (high-sensitivity)",
        "category": "inflammation",
        "aliases": ["crp", "c-reactive protein", "hs-crp", "hscrp"],
        "unit": "mg/L",
        "ref_low": 0, "ref_high": 3.0,
        "athlete_optimal_low": 0, "athlete_optimal_high": 1.0,
        "critical_low": 0, "critical_high": 10.0,
        "performance_note": "Persistently elevated CRP (>2 mg/L) can indicate "
                            "chronic inflammation or overtraining.",
    },

    # ─── Omega-3 ─────────────────────────────────────────────────────────────
    "omega3_index": {
        "label": "Omega-3 Index",
        "category": "fats",
        "aliases": ["omega-3 index", "omega 3 index", "epa+dha"],
        "unit": "%",
        "ref_low": 4, "ref_high": 12,
        "athlete_optimal_low": 8, "athlete_optimal_high": 12,
        "critical_low": 2, "critical_high": 16,
    },

    # ─── Amino acids / methylation ───────────────────────────────────────────
    "homocysteine": {
        "label": "Homocysteine",
        "category": "methylation",
        "aliases": ["homocysteine", "homocystein", "s-homocystein", "p-homocystein", "hcy"],
        "unit": "µmol/L",
        "ref_low": 0, "ref_high": 15,
        "athlete_optimal_low": 0, "athlete_optimal_high": 10,
        "critical_low": 0, "critical_high": 50,
        "performance_note": ">15 µmol/L → impaired methylation; B12/folate deficiency often causal.",
    },

    # ─── Complete blood count (CBC) ──────────────────────────────────────────
    "wbc": {
        "label": "White Blood Cells (WBC)",
        "category": "cbc",
        "aliases": ["wbc", "white blood cell", "leukocytes", "leukocyter", "b-leukocyter", "leucocytes"],
        "unit": "×10⁹/L",
        "ref_low": 4.0, "ref_high": 11.0,
        "athlete_optimal_low": 4.0, "athlete_optimal_high": 8.0,
        "critical_low": 2.5, "critical_high": 20.0,
    },
    "rbc": {
        "label": "Red Blood Cells (RBC)",
        "category": "cbc",
        "aliases": ["rbc", "red blood cell", "erythrocytes", "erytrocyter", "b-erytrocyter", "erythrocyte count"],
        "unit": "×10¹²/L",
        "ref_low": 4.2, "ref_high": 5.9,
        "athlete_optimal_low": 4.5, "athlete_optimal_high": 5.5,
        "critical_low": 3.0, "critical_high": 7.0,
        "sex_specific": {"female": (3.8, 5.2), "male": (4.5, 5.9)},
    },
    "platelets": {
        "label": "Platelets",
        "category": "cbc",
        "aliases": ["platelets", "thrombocytes", "thrombocyter", "trombocyter", "b-trombocyter", "plt"],
        "unit": "×10⁹/L",
        "ref_low": 150, "ref_high": 400,
        "athlete_optimal_low": 150, "athlete_optimal_high": 400,
        "critical_low": 50, "critical_high": 1000,
    },
    "mcv": {
        "label": "MCV",
        "category": "cbc",
        "aliases": ["mcv", "b-mcv", "mean corpuscular volume"],
        "unit": "fL",
        "ref_low": 80, "ref_high": 100,
        "athlete_optimal_low": 82, "athlete_optimal_high": 98,
        "critical_low": 70, "critical_high": 115,
    },
    "mch": {
        "label": "MCH",
        "category": "cbc",
        "aliases": ["mch", "b-mch", "mean corpuscular hemoglobin"],
        "unit": "pg",
        "ref_low": 27, "ref_high": 33,
        "athlete_optimal_low": 27, "athlete_optimal_high": 33,
        "critical_low": 20, "critical_high": 40,
    },
    "mchc": {
        "label": "MCHC",
        "category": "cbc",
        "aliases": ["mchc", "b-mchc", "mean corpuscular hemoglobin concentration"],
        "unit": "g/dL",
        "ref_low": 32, "ref_high": 36,
        "athlete_optimal_low": 32, "athlete_optimal_high": 36,
        "critical_low": 28, "critical_high": 38,
    },
    "esr": {
        "label": "ESR (Sedimentation Rate)",
        "category": "inflammation",
        "aliases": ["esr", "erythrocyte sedimentation", "sedimentation rate", "b-sr", "sr,", "b-sr,", "sänka"],
        "unit": "mm/h",
        "ref_low": 0, "ref_high": 20,
        "athlete_optimal_low": 0, "athlete_optimal_high": 15,
        "critical_low": 0, "critical_high": 100,
        "sex_specific": {"female": (0, 30), "male": (0, 20)},
    },
}


# ── Unit conversion table (target → factor from common alt units) ────────────
# Only conversions actually seen in lab reports.
UNIT_CONVERSIONS: dict[tuple[str, str], float] = {
    # ferritin µg/L == ng/mL (1:1 by definition) — handled in normalize_unit
    # B12 pmol/L → pg/mL: divide by 0.738
    ("pmol/L", "pg/mL"):     1 / 0.738,
    # folate nmol/L → ng/mL: divide by 2.265
    ("nmol/L", "ng/mL"):     1 / 2.265,
    # Vit D nmol/L → ng/mL: divide by 2.5
    # (handled per-marker below)
    # Glucose mmol/L → mg/dL: × 18.0156
    ("mmol/L", "mg/dL"):     18.0156,
    # Cholesterol mmol/L → mg/dL: × 38.67
}

# Marker-specific conversions when the generic table is ambiguous
SPECIFIC_CONVERSIONS: dict[str, dict[str, float]] = {
    "vitamin_d":  {"nmol/L": 1 / 2.5},
    "ferritin":   {"µg/L": 1.0, "ug/L": 1.0, "mcg/L": 1.0},
    "vitamin_b12": {"pmol/L": 1 / 0.738},
    "folate":     {"nmol/L": 1 / 2.265},
    "glucose_fasting": {"mmol/L": 18.0156},
    "total_cholesterol": {"mmol/L": 38.67},
    "ldl":        {"mmol/L": 38.67},
    "hdl":        {"mmol/L": 38.67},
    "triglycerides": {"mmol/L": 88.57},
    "magnesium":  {"mmol/L": 2.43},  # × 2.43 → mg/dL
    "calcium":    {"mmol/L": 4.008},
    # Swedish / SI units commonly seen on Nordic lab reports
    "hemoglobin": {"g/L": 0.1},                       # 145 g/L → 14.5 g/dL
    "serum_iron": {"µmol/L": 5.585, "umol/L": 5.585}, # Fe MW 55.85 → µg/dL
    "sodium":     {"mmol/L": 1.0},                    # already canonical, explicit no-op
    "potassium":  {"mmol/L": 1.0},
    "ck":         {"µkat/L": 60.0, "ukat/L": 60.0},   # 1 µkat/L = 60 U/L
    "ldh":        {"µkat/L": 60.0, "ukat/L": 60.0},
    "ast":        {"µkat/L": 60.0, "ukat/L": 60.0},
    "alt":        {"µkat/L": 60.0, "ukat/L": 60.0},
    "creatinine": {"µmol/L": 1 / 88.4, "umol/L": 1 / 88.4},  # → mg/dL
    "urea":       {"mmol/L": 2.801},                  # urea mmol/L → mg/dL (BUN)
    "hematocrit": {"L/L": 100.0, "ratio": 100.0},     # 0.45 L/L → 45 %
    # Thyroid — Swedish mE/L ≡ mIU/L for TSH
    "tsh":        {"mE/L": 1.0},
    # Free T4: pmol/L → ng/dL (T4 MW 776.87 g/mol)
    "free_t4":    {"pmol/L": 0.0777},
    # CBC cell counts: 10E9/L and 10E12/L arrive from UNIT_RE as-is
    "wbc":        {"10E9/L": 1.0,  "10e9/l": 1.0},
    "platelets":  {"10E9/L": 1.0,  "10e9/l": 1.0},
    "rbc":        {"10E12/L": 1.0, "10e12/l": 1.0},
    # MCHC in Swedish labs often reported as g/L
    "mchc":       {"g/L": 0.1},
}


def normalize_unit(marker_key: str, value: float, unit: Optional[str]) -> tuple[float, str]:
    """Convert (value, unit) to the marker's canonical unit. Returns (value, canonical_unit)."""
    if marker_key not in MARKERS:
        return value, unit or ""
    target = MARKERS[marker_key]["unit"]
    u = (unit or "").strip()

    # Special: hematocrit as decimal fraction — ANY value ≤ 1.0 must be a fraction.
    # 0.44 L/L or 0.44 % are both physically impossible as a real %; multiply by 100.
    if marker_key == "hematocrit" and value <= 1.0:
        return round(value * 100.0, 1), target

    # Special: HbA1c IFCC (mmol/mol) → NGSP (%) — nonlinear, can't use simple factor
    if marker_key == "hba1c" and u.lower() == "mmol/mol":
        return round(value * 0.0915 + 2.15, 2), target

    if not u or u.lower() == target.lower():
        return value, target
    spec = SPECIFIC_CONVERSIONS.get(marker_key, {})
    if u in spec:
        return value * spec[u], target
    if (u, target) in UNIT_CONVERSIONS:
        return value * UNIT_CONVERSIONS[(u, target)], target
    # unknown — keep as-is, downstream will mark unknown status
    return value, u


def status_for_value(
    marker_key: str, value: float, sex: Optional[str] = None
) -> str:
    """Return one of: critical_low, low, suboptimal, optimal, high, critical_high, unknown."""
    m = MARKERS.get(marker_key)
    if not m:
        return "unknown"

    # Use sex-specific range if available
    ref_low, ref_high = m["ref_low"], m["ref_high"]
    if sex and "sex_specific" in m:
        sx = sex.lower()
        if sx in m["sex_specific"]:
            ref_low, ref_high = m["sex_specific"][sx]

    if value < m["critical_low"]:  return "critical_low"
    if value > m["critical_high"]: return "critical_high"
    if value < ref_low:            return "low"
    if value > ref_high:           return "high"
    if m["athlete_optimal_low"] <= value <= m["athlete_optimal_high"]:
        return "optimal"
    return "suboptimal"


def find_marker_key(text: str) -> Optional[str]:
    """Find a marker key whose alias appears in `text` (case-insensitive)."""
    t = text.lower()
    # Match longer aliases first to avoid e.g. "iron" matching before "serum iron"
    candidates: list[tuple[int, str]] = []
    for key, m in MARKERS.items():
        for alias in m["aliases"]:
            if alias.lower() in t:
                candidates.append((len(alias), key))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]
