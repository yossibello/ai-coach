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

from app.nutrition.markers import MARKERS, normalize_unit, status_for_value


# Use Claude Haiku — fastest + cheapest vision model that handles labs well.
CLAUDE_MODEL = os.getenv("ANTHROPIC_VISION_MODEL", "claude-3-5-haiku-20241022")
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

_PROMPT = f"""You are a medical lab-report extractor. Look at the attached scanned blood-test pages and return ALL recognisable lab values as JSON.

Output ONLY valid JSON in this exact shape (no markdown fences, no commentary):
{{
  "lab_name": "string or null",
  "test_date": "YYYY-MM-DD or null",
  "markers": [
    {{"key": "<one of the allowed keys below>", "value": <number>, "unit": "<unit string>", "ref_low": <number or null>, "ref_high": <number or null>}}
  ]
}}

Allowed marker keys (use ONLY these; omit any marker you cannot confidently match):
{", ".join(_VALID_MARKER_KEYS)}

Rules:
- Use the EXACT key from the allowed list. Reports may be in English, Swedish, German, etc. — match by meaning.
  Examples: "S-Ferritin"→ferritin, "B-Hemoglobin"→hemoglobin, "S-25-OH-Vitamin D"→vitamin_d,
  "S-Kobalamin"→vitamin_b12, "S-Folat"→folate, "P-Glukos"→glucose_fasting,
  "S-Järn"→serum_iron, "S-Kreatinin"→creatinine, "S-ASAT"→ast, "S-ALAT"→alt,
  "S-Kalium"→potassium, "S-Natrium"→sodium, "S-Kolesterol"→total_cholesterol,
  "S-Magnesium"→magnesium, "EVF"→hematocrit, "S-Testosteron"→testosterone_total,
  "S-Kortisol"→cortisol, "Kreatinkinas/CK"→ck, "S-LD"→ldh.
- Preserve the unit AS WRITTEN ON THE REPORT (e.g. "g/L", "µmol/L", "µkat/L", "nmol/L", "mmol/L").
- value/ref_low/ref_high must be numbers (use a dot as the decimal separator even if the report uses a comma).
- If a reference range is shown next to the value, include it; otherwise use null.
- If a value is missing, illegible, or ambiguous, OMIT that marker entirely.
- Ignore patient demographics, comments, and any value not in the allowed list.
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

        try:
            value_norm, unit_norm = normalize_unit(key, value, unit)
        except Exception:
            value_norm, unit_norm = value, (unit or "")

        m_def = MARKERS[key]
        ref_low = entry.get("ref_low")
        ref_high = entry.get("ref_high")
        if ref_low is None:
            ref_low = m_def["ref_low"]
        if ref_high is None:
            ref_high = m_def["ref_high"]
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
