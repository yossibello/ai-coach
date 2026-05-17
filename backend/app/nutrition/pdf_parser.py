"""
Blood-test PDF parser.

Approach: text-extract with pdfplumber (fast, no OCR), then regex-scan each line
for known marker aliases. For each line that contains a marker alias, look for
a number + unit token nearby. Heuristic, intentionally conservative — emits a
per-marker confidence score so the UI can flag uncertain extractions for review.

If pdfplumber returns no text (image-only PDF), we surface a clear error and
the user falls back to manual entry.

Returns:
  {
    "lab_name":          Optional[str],
    "test_date":         Optional[ISO date],
    "markers":           {marker_key: {value, unit, ref_low, ref_high}},
    "parser_version":    "v1",
    "parser_confidence": float in [0,1],   # avg per-marker confidence
    "raw_text_preview":  str (first 500 chars, for debugging),
    "warnings":          list[str],
  }
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Optional

from app.nutrition.markers import (
    MARKERS,
    find_marker_key,
    normalize_unit,
    status_for_value,
)


PARSER_VERSION = "v1"

# Look for "value unit" pattern: number, optional unit, optional reference range
# Examples it should catch:
#   "Ferritin                  35   ng/mL    30 - 400"
#   "Ferritin: 35 ng/mL (30-400)"
#   "FERRITIN .......... 35.2 ng/mL"
NUMBER_RE = re.compile(r"(?<![A-Za-z\d])(\d+(?:[.,]\d+)?)(?![A-Za-z])")
UNIT_RE = re.compile(
    r"(ng/mL|µg/dL|ug/dL|mcg/dL|mg/dL|g/dL|µg/L|ug/L|mcg/L|g/L|"
    r"pg/mL|mIU/L|µIU/mL|uIU/mL|nmol/L|pmol/L|µmol/L|umol/L|mmol/mol|mmol/L|"
    r"µkat/L|ukat/L|mkat/L|"
    r"10E12/L|10E9/L|fL|mE/L|mm/h|"
    r"U/L|IU/L|%|mL/min/1\.73m²)",
    re.IGNORECASE,
)
RANGE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[-–to]+\s*(\d+(?:[.,]\d+)?)"
)
DATE_RE = re.compile(
    r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b|"
    r"\b(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})\b"
)


def parse_blood_test_pdf(pdf_bytes: bytes, sex: Optional[str] = None) -> dict:
    """Main entry point. Accepts raw PDF bytes."""
    try:
        import pdfplumber  # lazy import — only needed when actually parsing
    except ImportError:
        return _error_result("pdfplumber not installed — run: pip install pdfplumber")

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "\n".join(
                (page.extract_text() or "") for page in pdf.pages
            )
    except Exception as e:
        return _error_result(f"PDF could not be opened: {e}")

    # First pass: regex-based text extraction (fast, free, deterministic)
    text_result = _parse_text(full_text, sex) if full_text.strip() else None

    # Decide whether to fall back to vision-LLM extraction:
    #   - no text in the PDF (image-only scan), OR
    #   - regex parser found < 3 markers (likely unsupported lab format)
    needs_vision = (
        text_result is None
        or len(text_result.get("markers", {})) < 3
    )

    if needs_vision:
        try:
            from app.nutrition import vision_extractor
        except ImportError:
            vision_extractor = None  # type: ignore

        if vision_extractor and vision_extractor.is_available():
            vision_result = vision_extractor.extract_via_vision(pdf_bytes, sex=sex)
            text_count = len(text_result.get("markers", {})) if text_result else 0
            vision_count = len(vision_result.get("markers", {}))
            # Always prefer vision when there was no text at all (image PDF).
            # Also prefer vision when it found more markers than the regex pass.
            if text_result is None or vision_count > text_count:
                if text_result and text_result.get("warnings"):
                    vision_result.setdefault("warnings", []).extend(text_result["warnings"])
                return vision_result
        elif text_result is None:
            # No text AND no vision available — surface clear actionable error
            return _error_result(
                "This PDF contains no extractable text (image-only / scanned) and the "
                "vision-extraction fallback is not configured. Either: "
                "(1) re-export the PDF as a text-PDF from your lab portal, "
                "(2) enable Claude vision (set ANTHROPIC_API_KEY env var and "
                "`pip install anthropic pypdfium2`), "
                "or (3) enter values manually."
            )

    if text_result is None:
        return _error_result(
            "No text extracted from PDF — file is likely image-only / scanned. "
            "Please re-export as text-PDF or enter values manually."
        )

    return text_result


def _error_result(msg: str) -> dict:
    return {
        "lab_name": None,
        "test_date": None,
        "markers": {},
        "parser_version": PARSER_VERSION,
        "parser_confidence": 0.0,
        "raw_text_preview": "",
        "warnings": [msg],
    }


def _parse_text(text: str, sex: Optional[str]) -> dict:
    warnings: list[str] = []
    markers_out: dict[str, dict] = {}
    confidences: list[float] = []

    test_date = _extract_date(text)
    lab_name = _extract_lab_name(text)

    # Process line-by-line — most lab reports have one marker per line
    for line in text.splitlines():
        line_clean = line.strip()
        if not line_clean or len(line_clean) < 4:
            continue

        marker_key = find_marker_key(line_clean)
        if not marker_key:
            continue

        # Avoid double-recording — keep first occurrence (top of report = most recent)
        if marker_key in markers_out:
            continue

        parsed = _extract_value_unit_range(line_clean)
        if not parsed:
            continue
        value, unit, rng_low, rng_high, conf = parsed

        try:
            value_norm, unit_norm = normalize_unit(marker_key, value, unit)
        except Exception:
            value_norm, unit_norm = value, (unit or "")

        m_def = MARKERS[marker_key]
        # Convert the extracted ref range with the same unit as the value.
        # Only use PDF ref values if BOTH are present — mixing PDF + canonical
        # produces Frankenstein ranges (e.g. "0.6–100" for creatinine).
        if rng_low is not None and rng_high is not None:
            try:
                rng_low, _ = normalize_unit(marker_key, rng_low, unit)
                rng_high, _ = normalize_unit(marker_key, rng_high, unit)
                rng_low = round(rng_low, 3)
                rng_high = round(rng_high, 3)
            except Exception:
                rng_low, rng_high = None, None
        if rng_low is not None and rng_high is not None:
            ref_low, ref_high = rng_low, rng_high
        else:
            ref_low, ref_high = m_def["ref_low"], m_def["ref_high"]
        if sex and "sex_specific" in m_def and sex.lower() in m_def["sex_specific"]:
            ref_low, ref_high = m_def["sex_specific"][sex.lower()]

        status = status_for_value(marker_key, value_norm, sex)

        markers_out[marker_key] = {
            "value":    round(value_norm, 4),
            "unit":     unit_norm,
            "ref_low":  ref_low,
            "ref_high": ref_high,
            "status":   status,
            "label":    m_def["label"],
            "category": m_def["category"],
            "_confidence": round(conf, 3),
            "_raw_line": line_clean[:200],
        }
        confidences.append(conf)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    if not markers_out:
        warnings.append(
            "No recognized markers found. The lab format may be unsupported — "
            "please enter values manually."
        )

    return {
        "lab_name":          lab_name,
        "test_date":         test_date,
        "markers":           markers_out,
        "parser_version":    PARSER_VERSION,
        "parser_confidence": round(avg_conf, 3),
        "raw_text_preview":  text[:500],
        "warnings":          warnings,
    }


def _extract_value_unit_range(line: str):
    """Extract (value, unit, ref_low, ref_high, confidence) from a line."""
    nums = NUMBER_RE.findall(line)
    if not nums:
        return None
    unit_match = UNIT_RE.search(line)
    range_match = RANGE_RE.search(line)

    confidence = 0.4

    # Prefer the first number that isn't part of a range pair when picking the value.
    range_pair = None
    if range_match:
        range_pair = (range_match.group(1), range_match.group(2))

    value: Optional[float] = None
    for n in nums:
        if range_pair and n in range_pair:
            continue
        try:
            value = float(n.replace(",", "."))
            break
        except ValueError:
            continue
    if value is None:
        try:
            value = float(nums[0].replace(",", "."))
        except ValueError:
            return None

    unit = unit_match.group(0) if unit_match else None
    if unit:
        confidence += 0.25

    rng_low = rng_high = None
    if range_match:
        try:
            rng_low = float(range_match.group(1).replace(",", "."))
            rng_high = float(range_match.group(2).replace(",", "."))
            confidence += 0.25
        except ValueError:
            pass

    # Sanity: value should be in a plausible range for SOME marker
    if value <= 0 or value > 1e6:
        return None

    return value, unit, rng_low, rng_high, min(confidence, 1.0)


def _extract_date(text: str) -> Optional[str]:
    """Find first plausible date in the text. Returns ISO yyyy-mm-dd."""
    head = text[:2000]   # date almost always near top
    for m in DATE_RE.finditer(head):
        groups = [g for g in m.groups() if g]
        if len(groups) != 3:
            continue
        a, b, c = groups
        # Year-first?
        if len(a) == 4:
            yyyy, mm, dd = a, b, c
        else:
            dd, mm, yyyy = a, b, c
            if len(yyyy) == 2:
                yyyy = ("20" + yyyy) if int(yyyy) < 70 else ("19" + yyyy)
        try:
            return datetime(int(yyyy), int(mm), int(dd)).date().isoformat()
        except ValueError:
            continue
    return None


def _extract_lab_name(text: str) -> Optional[str]:
    """Heuristic: first ALL-CAPS line in the first page that looks like a name."""
    for line in text.splitlines()[:10]:
        s = line.strip()
        if 4 < len(s) < 60 and s.upper() == s and any(c.isalpha() for c in s):
            return s.title()
    return None
