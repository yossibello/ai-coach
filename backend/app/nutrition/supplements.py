"""
Supplement catalog with evidence grades, doses, timing, and contraindications.

Evidence grades follow the AIS Sport Supplement Framework:
  A — Strong evidence in athletes (use protocol-driven)
  B — Some evidence; consider in specific scenarios
  C — Mixed/weak evidence; experimental
  D — Banned, illegal, or potentially harmful

Each entry produces a recommendation block when the engine's depletion or
context score for that supplement crosses a trigger threshold.
"""
from __future__ import annotations


# Each catalog entry:
#   key, label, evidence_grade, category,
#   default_dose, unit, frequency, timing, duration,
#   rationale (template), citations, contraindications, warnings,
#   trigger_signals (which depletion/context signals activate it)
SUPPLEMENTS: dict[str, dict] = {

    # ──────────────────────────────────────────────────────────────────────────
    # GROUP A — Strong evidence for endurance performance
    # ──────────────────────────────────────────────────────────────────────────

    "caffeine": {
        "label": "Caffeine",
        "category": "performance",
        "evidence_grade": "A",
        "default_dose": 3.0,                       # mg per kg body mass
        "dose_unit": "mg/kg",
        "frequency": "pre-session",
        "timing": "30-60 min before key sessions or races",
        "duration": "as needed",
        "rationale": "3-6 mg/kg ~45 min pre-event reliably improves endurance "
                     "performance by ~3-7% via reduced perceived exertion and "
                     "central drive (ISSN 2021).",
        "citations": ["ISSN Position Stand: Caffeine and Exercise Performance (2021)",
                      "Guest et al. JISSN 2021"],
        "contraindications": ["pregnancy", "uncontrolled hypertension", "anxiety_disorder"],
        "warnings": ["Avoid within 6h of bedtime — disrupts sleep / recovery.",
                     "Cycle off 7-10 days every 6-8 weeks to restore sensitivity."],
        "triggers": ["pre_key_session"],
    },

    "creatine_monohydrate": {
        "label": "Creatine Monohydrate",
        "category": "performance",
        "evidence_grade": "A",
        "default_dose": 5.0,
        "dose_unit": "g",
        "frequency": "daily",
        "timing": "any time of day, with carbohydrate for absorption",
        "duration": "ongoing",
        "rationale": "Daily 3-5 g creatine improves repeated high-intensity efforts, "
                     "supports muscle mass, and shows growing evidence for cognition "
                     "and recovery in endurance athletes (ISSN 2017).",
        "citations": ["ISSN Position Stand: Creatine Supplementation (2017)",
                      "Kreider et al. JISSN 2017"],
        "contraindications": ["chronic_kidney_disease"],
        "warnings": ["Expect 1-2 kg water-weight gain in first month."],
        "triggers": ["high_intensity_focus", "vegetarian_diet", "masters_athlete"],
    },

    "beetroot_nitrate": {
        "label": "Beetroot Juice / Dietary Nitrate",
        "category": "performance",
        "evidence_grade": "A",
        "default_dose": 6.4,                # mmol nitrate (~500 ml beet juice)
        "dose_unit": "mmol NO₃⁻",
        "frequency": "pre-session, 3-day load before A-events",
        "timing": "2-3 h pre-event for acute; daily for 3-6 days for chronic",
        "duration": "3-6 days pre-event",
        "rationale": "Reduces O₂ cost of submaximal exercise by ~3% and improves "
                     "TT performance by 1-3%. Most effective in <30-min efforts.",
        "citations": ["IOC Consensus 2018", "Jones AM. Sports Med 2014"],
        "contraindications": [],
        "warnings": ["Avoid antibacterial mouthwash within 6h — kills oral nitrate-"
                     "reducing bacteria and abolishes the effect."],
        "triggers": ["pre_a_event"],
    },

    "beta_alanine": {
        "label": "Beta-Alanine",
        "category": "performance",
        "evidence_grade": "A",
        "default_dose": 4.0,
        "dose_unit": "g",
        "frequency": "daily, split 2 × 2 g",
        "timing": "with meals to reduce paresthesia",
        "duration": "10-12 weeks loading, then maintain 1.5-2 g/day",
        "rationale": "Increases muscle carnosine, buffering H⁺ during 1-10 min max "
                     "efforts. Best for VO₂max intervals and crit-style racing.",
        "citations": ["ISSN Position Stand: Beta-Alanine (2015)"],
        "contraindications": [],
        "warnings": ["Causes harmless paresthesia (skin tingling) — split doses help."],
        "triggers": ["vo2max_focus", "crit_race_focus"],
    },

    "sodium_bicarbonate": {
        "label": "Sodium Bicarbonate",
        "category": "performance",
        "evidence_grade": "A",
        "default_dose": 0.3,                # g per kg body mass
        "dose_unit": "g/kg",
        "frequency": "pre-event (acute)",
        "timing": "60-180 min pre-event; split doses reduce GI distress",
        "duration": "single use; rehearse in training first",
        "rationale": "Extracellular buffering improves repeated 1-7 min max efforts "
                     "by ~2%. Most useful for crits and short TTs.",
        "citations": ["IOC Consensus 2018", "ISSN 2021"],
        "contraindications": ["hypertension", "salt_sensitive"],
        "warnings": ["GI distress is common — ALWAYS test in training first.",
                     "High sodium load — not for hypertensive athletes."],
        "triggers": ["short_max_effort_event"],
    },

    "iron": {
        "label": "Iron (bisglycinate or sulfate)",
        "category": "deficiency_correction",
        "evidence_grade": "A",
        "default_dose": 60,                  # mg elemental iron, alternate days
        "dose_unit": "mg elemental",
        "frequency": "alternate days (every other day)",
        "timing": "morning, on empty stomach with vitamin C, away from coffee/tea/dairy",
        "duration": "8-12 weeks then re-test ferritin",
        "rationale": "Iron deficiency without anemia (ferritin <35 ng/mL) impairs "
                     "endurance performance and time to fatigue. Alternate-day dosing "
                     "improves absorption vs daily (Stoffel 2017).",
        "citations": ["Sim et al. Eur J Appl Physiol 2019",
                      "Stoffel et al. Lancet Haematol 2017",
                      "ISSN Position Stand: Iron (2014)"],
        "contraindications": ["hemochromatosis", "iron_overload", "ferritin_high"],
        "warnings": ["NEVER supplement iron without confirmed low ferritin — iron "
                     "overload is harmful and irreversible.",
                     "Re-test ferritin after 8-12 weeks."],
        "triggers": ["iron_deficient", "iron_subclinical_low"],
    },

    "vitamin_d3": {
        "label": "Vitamin D3 (cholecalciferol)",
        "category": "deficiency_correction",
        "evidence_grade": "A",
        "default_dose": 2000,
        "dose_unit": "IU",
        "frequency": "daily",
        "timing": "with the largest fat-containing meal",
        "duration": "3 months then re-test 25(OH)D",
        "rationale": "<30 ng/mL 25(OH)D impairs muscle function, immunity, and "
                     "bone remodeling. Most athletes living above 35° latitude or "
                     "training indoors are deficient in winter.",
        "citations": ["Owens et al. Sports Med 2018", "ISSN Position Stand: Vit D 2018"],
        "contraindications": ["hypercalcemia", "sarcoidosis"],
        "warnings": ["Re-test after 12 weeks. Don't exceed 4000 IU/day without "
                     "physician guidance."],
        "triggers": ["vitamin_d_deficient", "vitamin_d_low", "winter_indoor_training"],
    },

    "electrolytes": {
        "label": "Electrolyte Mix (Na/K/Mg)",
        "category": "performance",
        "evidence_grade": "A",
        "default_dose": 500,                  # mg sodium per L fluid
        "dose_unit": "mg Na/L",
        "frequency": "during sessions >75 min or in heat",
        "timing": "sip 500-800 ml/h during exercise",
        "duration": "session-by-session",
        "rationale": "Sweat sodium losses average 0.5-1.5 g/L; replacement prevents "
                     "hyponatremia and cramping in long, hot, or salty-sweater sessions.",
        "citations": ["IOC Consensus 2018"],
        "contraindications": ["hypertension"],
        "warnings": ["Heavy sweaters may need 1000+ mg Na/L; light sweaters less."],
        "triggers": ["hot_climate", "long_session", "high_weekly_volume"],
    },

    "carbohydrate_drink": {
        "label": "Carbohydrate Drink (during session)",
        "category": "performance",
        "evidence_grade": "A",
        "default_dose": 60,
        "dose_unit": "g/h",
        "frequency": "during sessions >90 min",
        "timing": "30-90 g/h during, depending on duration & intensity",
        "duration": "session-by-session",
        "rationale": "Maintains blood glucose and spares muscle glycogen. >2.5 h "
                     "events benefit from 90 g/h using glucose+fructose blends.",
        "citations": ["IOC Consensus 2018", "Burke et al. J Sports Sci 2019"],
        "contraindications": ["diabetes_t1_uncontrolled"],
        "warnings": [],
        "triggers": ["long_session"],
    },

    "whey_protein": {
        "label": "Whey Protein (or pea protein)",
        "category": "recovery",
        "evidence_grade": "A",
        "default_dose": 25,
        "dose_unit": "g",
        "frequency": "post-session + spread across day",
        "timing": "within 1 h post-session; total intake 1.4-2.0 g/kg/day",
        "duration": "ongoing",
        "rationale": "Whey provides high leucine for MPS. Endurance athletes need "
                     "1.4-1.8 g/kg/day total protein, often higher than diet alone.",
        "citations": ["ISSN Position Stand: Protein and Exercise (2017)"],
        "contraindications": ["lactose_intolerance"],
        "warnings": [],
        "triggers": ["high_weekly_volume", "muscle_damage_high"],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # GROUP B — Some evidence; situational
    # ──────────────────────────────────────────────────────────────────────────

    "magnesium_glycinate": {
        "label": "Magnesium Glycinate",
        "category": "deficiency_correction",
        "evidence_grade": "B",
        "default_dose": 300,
        "dose_unit": "mg elemental",
        "frequency": "daily",
        "timing": "evening, with food",
        "duration": "ongoing",
        "rationale": "Heavy training increases urinary and sweat Mg losses. "
                     "Glycinate form is well-absorbed and not laxative. May aid "
                     "sleep quality.",
        "citations": ["Volpe SL. Adv Nutr 2015"],
        "contraindications": ["chronic_kidney_disease"],
        "warnings": ["Avoid magnesium oxide — poor absorption, laxative effect."],
        "triggers": ["magnesium_low", "high_weekly_volume", "poor_sleep"],
    },

    "omega3": {
        "label": "Omega-3 (EPA+DHA)",
        "category": "anti-inflammatory",
        "evidence_grade": "B",
        "default_dose": 2000,                 # combined EPA+DHA
        "dose_unit": "mg EPA+DHA",
        "frequency": "daily",
        "timing": "with a fat-containing meal",
        "duration": "ongoing",
        "rationale": "Reduces exercise-induced muscle inflammation and DOMS, "
                     "supports cardiovascular health and brain function. Targets "
                     "Omega-3 Index of 8-12%.",
        "citations": ["IOC Consensus 2018", "Heileson Sports Med 2023"],
        "contraindications": ["bleeding_disorder", "anticoagulant_therapy"],
        "warnings": ["Choose IFOS-tested brands to avoid heavy metals."],
        "triggers": ["omega3_low", "high_inflammation", "high_weekly_volume"],
    },

    "vitamin_b12_methylcobalamin": {
        "label": "Vitamin B12 (methylcobalamin)",
        "category": "deficiency_correction",
        "evidence_grade": "B",
        "default_dose": 1000,
        "dose_unit": "µg",
        "frequency": "daily",
        "timing": "any time, sublingual or oral",
        "duration": "8-12 weeks then re-test",
        "rationale": "Vegans, vegetarians, and PPI users are at high risk. "
                     "Deficiency causes fatigue, macrocytic anemia, and neuropathy.",
        "citations": ["NIH ODS Vitamin B12 Fact Sheet"],
        "contraindications": [],
        "warnings": [],
        "triggers": ["b12_low", "vegan_diet", "vegetarian_diet"],
    },

    "zinc_glycinate": {
        "label": "Zinc Glycinate",
        "category": "deficiency_correction",
        "evidence_grade": "B",
        "default_dose": 15,
        "dose_unit": "mg elemental",
        "frequency": "daily",
        "timing": "with food, away from iron and calcium",
        "duration": "8-12 weeks then re-test",
        "rationale": "Sweat losses can be significant; supports immune function and "
                     "testosterone production.",
        "citations": ["NIH ODS Zinc Fact Sheet"],
        "contraindications": [],
        "warnings": ["Don't exceed 40 mg/day long-term — interferes with copper."],
        "triggers": ["zinc_low", "frequent_illness"],
    },

    "tart_cherry": {
        "label": "Tart Cherry Extract",
        "category": "recovery",
        "evidence_grade": "B",
        "default_dose": 480,
        "dose_unit": "mg extract (or 30 ml concentrate)",
        "frequency": "daily during heavy blocks or pre-event",
        "timing": "evening (may aid sleep) or post-session",
        "duration": "5 days pre-event through 48 h post",
        "rationale": "Anthocyanins reduce DOMS and inflammation post-eccentric "
                     "exercise. Some evidence for sleep onset.",
        "citations": ["Bell et al. Nutrients 2014"],
        "contraindications": [],
        "warnings": ["Avoid chronic high-dose during base — may blunt adaptation."],
        "triggers": ["pre_a_event", "muscle_damage_high"],
    },

    "collagen_vitC": {
        "label": "Collagen + Vitamin C (pre-tendon work)",
        "category": "tissue",
        "evidence_grade": "B",
        "default_dose": 15,
        "dose_unit": "g collagen + 50 mg vit C",
        "frequency": "pre-tendon-loading sessions",
        "timing": "60 min before activity",
        "duration": "ongoing if rehabbing tendon issue",
        "rationale": "15 g collagen + vitamin C 60 min pre-loading doubles collagen "
                     "synthesis at tendon — useful for tendinopathy rehab.",
        "citations": ["Shaw et al. Am J Clin Nutr 2017"],
        "contraindications": [],
        "warnings": [],
        "triggers": ["tendon_rehab"],
    },

    "probiotics": {
        "label": "Probiotic (Lactobacillus + Bifidobacterium)",
        "category": "immunity",
        "evidence_grade": "B",
        "default_dose": 1e10,
        "dose_unit": "CFU",
        "frequency": "daily",
        "timing": "morning, away from antibiotics",
        "duration": "during heavy training blocks and travel",
        "rationale": "May reduce URTI incidence and severity in heavy-training "
                     "athletes. Strain-specific effects.",
        "citations": ["IOC Consensus 2018", "Pyne et al. Nutrients 2015"],
        "contraindications": ["immunocompromised"],
        "warnings": [],
        "triggers": ["frequent_illness", "international_travel"],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # GROUP C — Mixed evidence; experimental
    # ──────────────────────────────────────────────────────────────────────────

    "ashwagandha": {
        "label": "Ashwagandha (KSM-66)",
        "category": "stress",
        "evidence_grade": "C",
        "default_dose": 600,
        "dose_unit": "mg",
        "frequency": "daily",
        "timing": "split AM/PM with food",
        "duration": "8-12 weeks",
        "rationale": "Some evidence for reduced cortisol and improved VO₂max in "
                     "untrained subjects; weaker in trained endurance athletes.",
        "citations": ["Sandhu et al. Int J Ayurveda Res 2010"],
        "contraindications": ["pregnancy", "thyroid_disorder", "autoimmune_disease"],
        "warnings": [],
        "triggers": ["high_cortisol", "perceived_stress_high"],
    },

    "hmb": {
        "label": "HMB (β-Hydroxy β-Methylbutyrate)",
        "category": "recovery",
        "evidence_grade": "C",
        "default_dose": 3,
        "dose_unit": "g",
        "frequency": "daily, split 3 doses",
        "timing": "with meals",
        "duration": "during heavy/return-to-training blocks",
        "rationale": "May reduce muscle damage during heavy training or after layoff. "
                     "Effect smaller in trained athletes.",
        "citations": ["ISSN Position Stand: HMB 2013"],
        "contraindications": [],
        "warnings": [],
        "triggers": ["return_from_layoff", "very_high_volume"],
    },
}


# ── Anti-supplement warnings (things to AVOID during base/build) ────────────
# These are returned as warnings, not stack items.
ANTI_SUPPLEMENTS = {
    "high_dose_antioxidants": {
        "trigger": "always",
        "applies_to": ["vitamin C >1000 mg/d chronic", "vitamin E >400 IU/d chronic"],
        "message": "AVOID chronic high-dose vitamin C/E during training — multiple "
                   "RCTs (Paulsen 2014, Morrison 2015) show they BLUNT mitochondrial "
                   "biogenesis and training adaptation. Acute use post-illness only.",
        "citations": ["Paulsen et al. J Physiol 2014", "Morrison et al. Free Radic Biol Med 2015"],
    },
    "nsaids_chronic": {
        "trigger": "always",
        "applies_to": ["ibuprofen, naproxen — chronic use"],
        "message": "AVOID chronic NSAIDs during training blocks — impair muscle "
                   "and tendon adaptation, increase GI/kidney risk during long efforts.",
        "citations": ["Lilja et al. Acta Physiol 2018"],
    },
    "iron_unverified": {
        "trigger": "no_blood_test_with_high_load",
        "applies_to": ["iron supplements without confirmed deficiency"],
        "message": "Do NOT take iron without a recent ferritin test confirming "
                   "deficiency. Iron overload (hemochromatosis) is serious and "
                   "irreversible.",
        "citations": ["Sim 2019"],
    },
}
