"""
Anthropic Claude vision-based blood test extractor.

Used as a fallback when pdfplumber returns no extractable text (image-only
scanned PDFs). Sends each page as a base64-encoded image to Claude and asks
for structured JSON containing the recognised lab markers.

Requires ANTHROPIC_API_KEY env var. The `anthropic` Python package is imported
lazily so the rest of the app keeps working when the dep / key is missing.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Optional

from app.nutrition.markers import (
    MARKERS, SPECIFIC_CONVERSIONS, UNIT_CONVERSIONS,
    normalize_unit, status_for_value,
)


# Use Claude Haiku — fastest + cheapest vision model that handles labs well.
# Sonnet is more accurate than Haiku for medical label parsing — worth the cost.
CLAUDE_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
PARSER_VERSION_VISION = "vision-v1"


def is_available() -> bool:
    """True if anthropic SDK is installed AND ANTHROPIC_API_KEY is set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def extract_via_vision(pdf_bytes: bytes, sex: Optional[str] = None) -> dict:
    """Render PDF pages → PNG → Claude vision → structured markers dict.

    Returns the same shape as `_parse_text` in pdf_parser.py so it can be a
    drop-in fallback.
    """
    if not is_available():
        return _vision_error(
            "Vision fallback unavailable. Install with: pip install anthropic pypdfium2 "
            "and set ANTHROPIC_API_KEY env var."
        )

    try:
        page_images_b64 = _render_pdf_to_png_b64(pdf_bytes)
    except Exception as e:
        return _vision_error(f"Could not render PDF pages: {e}")

    if not page_images_b64:
        return _vision_error("PDF has no renderable pages.")

    try:
        raw = _ask_claude(page_images_b64)
    except Exception as e:
        return _vision_error(f"Claude API error: {e}")

    return _normalize_claude_output(raw, sex)


# ─── Internals ────────────────────────────────────────────────────────────────
def _render_pdf_to_png_b64(pdf_bytes: bytes, dpi: int = 200) -> list[str]:
    """Render every PDF page to base64-encoded PNG using pypdfium2 (no Poppler needed)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    out: list[str] = []
    scale = dpi / 72  # pdfium uses points
    for page in pdf:
        bitmap = page.render(scale=scale)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        # JPEG saves ~5x bandwidth vs PNG with no quality loss for text scans
        pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
        out.append(base64.standard_b64encode(buf.getvalue()).decode("ascii"))
    return out


# Build the marker key list once for the prompt
_VALID_MARKER_KEYS = list(MARKERS.keys())

# Pre-build valid source units per marker key (used for plausibility checks)
def _build_valid_units() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for key, m in MARKERS.items():
        canonical = m["unit"].lower()
        valid = {canonical, ""}
        for k in SPECIFIC_CONVERSIONS.get(key, {}):
            valid.add(k.lower())
        for (src, tgt) in UNIT_CONVERSIONS:
            if tgt.lower() == canonical:
                valid.add(src.lower())
        result[key] = valid
    return result

_VALID_UNITS_BY_KEY = _build_valid_units()

_PROMPT = f"""You are a precise medical lab-report extractor. Look at the attached scanned blood-test pages and return ALL recognisable lab values as JSON.

Output ONLY valid JSON in this exact shape (no markdown fences, no commentary):
{{
  "lab_name": "string or null",
  "test_date": "YYYY-MM-DD or null",
  "markers": [
    {{"key": "<one of the allowed keys below>", "value": <number>, "unit": "<unit string or null>", "ref_low": <number or null>, "ref_high": <number or null>}}
  ]
}}

Allowed marker keys (use ONLY these; omit any marker you cannot confidently match):
{", ".join(_VALID_MARKER_KEYS)}

CRITICAL MAPPING RULES — accuracy is required, this is medical data:
- Match ONLY by the marker label on the report, never by value or position.
- Each label maps to exactly one key. Common Swedish→key mappings:
  S-ALAT / ALAT → alt          S-ASAT / ASAT → ast
  S-Ferritin → ferritin        B-Hemoglobin → hemoglobin
  B-EVF / EVF → hematocrit     B-HbA1c → hba1c
  S-25-OH Vitamin D → vitamin_d  S-Kobalamin / B12 → vitamin_b12
  S-Folat → folate             S-Kreatinin → creatinine
  eGFR / S-eGFR / Estimerat GFR → egfr
  S-Natrium → sodium           S-Kalium → potassium
  S-TSH / S-TSH Tyreotropin → tsh
  S-fritt T4 / S-fritt T4 Tyroxin → free_t4
  S-Joniserat calcium → calcium_ionized
  S-Homocystein → homocysteine
  B-Leukocyter → wbc           B-Erytrocyter → rbc
  B-Trombocyter → platelets    B-MCV → mcv
  B-MCH → mch                  B-MCHC → mchc
  B-SR → esr                   S-Transferrinmättnad → transferrin_saturation
  S-Järn → serum_iron          S-Kolesterol → total_cholesterol
  S-Testosteron → testosterone_total  S-Kortisol → cortisol
- If you are not confident about a label→key match, OMIT that marker.
- Preserve the unit EXACTLY as written (e.g. "g/L", "µmol/L", "µkat/L", "nmol/L", "mmol/L", "mE/L", "mmol/mol").
- For values written as ">90" or "<0.1", extract the number only (90 or 0.1).
- value/ref_low/ref_high must be plain numbers (dot as decimal separator).
- If a reference range is shown, include it; otherwise null.
- OMIT any marker that is missing, illegible, or ambiguous.
"""


def _ask_claude(page_images_b64: list[str]) -> dict:
    import anthropic

    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env

    content: list[dict] = []
    for b64 in page_images_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": _PROMPT})

    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )

    # Concatenate text blocks
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    text = text.strip()

    # Strip ```json fences if Claude added them despite the instruction
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to extract the largest {...} block as a recovery
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Claude returned non-JSON: {text[:500]}") from e


def _normalize_claude_output(raw: dict, sex: Optional[str]) -> dict:
    """Convert Claude's JSON into the same shape `_parse_text` returns."""
    markers_out: dict[str, dict] = {}
    skipped: list[str] = []

    for entry in raw.get("markers", []) or []:
        key = entry.get("key")
        if key not in MARKERS:
            skipped.append(str(key))
            continue
        try:
            value = float(entry["value"])
        except (TypeError, ValueError, KeyError):
            continue
        unit = entry.get("unit") or None

        # Unit plausibility check: if the unit Claude returned can't be converted
        # to this marker's canonical unit, Claude likely assigned the wrong key.
        if unit:
            valid_units = _VALID_UNITS_BY_KEY.get(key, set())
            u_lower = unit.lower().strip()
            # Normalise µ variants for comparison
            u_norm = u_lower.replace("µ", "u").replace("μ", "u")
            valid_norm = {v.replace("µ", "u").replace("μ", "u") for v in valid_units}
            if u_norm not in valid_norm and valid_units != {""}:
                skipped.append(f"{key} (unit '{unit}' implausible — likely mislabeled by vision)")
                continue

        try:
            value_norm, unit_norm = normalize_unit(key, value, unit)
        except Exception:
            value_norm, unit_norm = value, (unit or "")

        # Value sanity check: if the normalised value is wildly below critical_low,
        # it's almost certainly a unit conversion error (e.g. hematocrit 0.44 as %).
        m_def = MARKERS[key]
        crit_low = m_def.get("critical_low", 0)
        crit_high = m_def.get("critical_high", 1e9)
        if crit_low > 0 and value_norm < crit_low * 0.05:
            skipped.append(f"{key} (value {value_norm} implausibly low — likely unit error)")
            continue
        if value_norm > crit_high * 20:
            skipped.append(f"{key} (value {value_norm} implausibly high — likely unit error)")
            continue

        m_def = MARKERS[key]
        pdf_ref_low = entry.get("ref_low")
        pdf_ref_high = entry.get("ref_high")

        # Convert ref range using the SAME unit as the value.
        # Only use PDF ref values if BOTH are present — mixing PDF (unconverted)
        # with MARKERS (canonical) produces Frankenstein ranges like "0.6–100".
        if pdf_ref_low is not None and pdf_ref_high is not None:
            try:
                ref_low, _ = normalize_unit(key, float(pdf_ref_low), unit)
                ref_high, _ = normalize_unit(key, float(pdf_ref_high), unit)
                ref_low = round(ref_low, 3)
                ref_high = round(ref_high, 3)
            except Exception:
                ref_low, ref_high = m_def["ref_low"], m_def["ref_high"]
        else:
            ref_low, ref_high = m_def["ref_low"], m_def["ref_high"]
        if sex and "sex_specific" in m_def and sex.lower() in m_def["sex_specific"]:
            ref_low, ref_high = m_def["sex_specific"][sex.lower()]

        markers_out[key] = {
            "value":    round(value_norm, 4),
            "unit":     unit_norm,
            "ref_low":  ref_low,
            "ref_high": ref_high,
            "status":   status_for_value(key, value_norm, sex),
            "label":    m_def["label"],
            "category": m_def["category"],
            "_confidence": 0.9,    # vision is high-confidence by default
            "_raw_line": f"[claude vision] {entry}",
        }

    warnings: list[str] = []
    if skipped:
        warnings.append(f"Vision skipped {len(skipped)} unrecognised marker key(s).")
    if not markers_out:
        warnings.append("Vision could not extract any markers from this PDF.")

    return {
        "lab_name":          raw.get("lab_name"),
        "test_date":         raw.get("test_date"),
        "markers":           markers_out,
        "parser_version":    PARSER_VERSION_VISION,
        "parser_confidence": 0.9 if markers_out else 0.0,
        "raw_text_preview":  f"[Extracted via Claude {CLAUDE_MODEL}]",
        "warnings":          warnings,
    }


def _vision_error(msg: str) -> dict:
    return {
        "lab_name": None,
        "test_date": None,
        "markers": {},
        "parser_version": PARSER_VERSION_VISION,
        "parser_confidence": 0.0,
        "raw_text_preview": "",
        "warnings": [msg],
    }
