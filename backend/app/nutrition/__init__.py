"""Nutrition / supplement engine (rule-based, Phase 1).

Sources used to author the rule tables in this package:
  • NIH Office of Dietary Supplements (ODS) — Health Professional Fact Sheets
  • IOC Consensus Statement: Dietary Supplements & High-Performance Athletes (2018)
  • ISSN Position Stands (Caffeine 2021, Creatine 2017, Beta-alanine 2015,
    Sodium bicarb 2021, Protein 2017, Iron 2014, Vitamin D 2018)
  • AIS Sport Supplement Framework (Group A/B/C/D classification)
  • Mayo Clinic & Cleveland Clinic reference ranges
  • Sim et al. 2019 — Iron considerations for the athlete (Eur J Appl Physiol)
  • Owens et al. 2018 — Vitamin D and the athlete (Sports Med)

ALL DOSES AND RANGES ARE FOR EDUCATED ATHLETIC ADULTS. NOT MEDICAL ADVICE.
The engine ALWAYS adds a "consult physician" warning and never recommends
anything contraindicated by the user's blood markers.
"""
