"""
Parse .gpx, .fit, .tcx files and return a dict of Activity fields.
"""
from __future__ import annotations

import math
from datetime import timezone
from typing import Any

import gpxpy
import fitparse

from app.services.hr_drift import compute_hr_drift_from_streams


def parse_activity_file(path: str, ext: str) -> dict[str, Any]:
    if ext == ".gpx":
        return _parse_gpx(path)
    elif ext == ".fit":
        return _parse_fit(path)
    elif ext == ".tcx":
        return _parse_tcx(path)
    raise ValueError(f"Unsupported extension: {ext}")


# ─── GPX ────────────────────────────────────────────────────────────────────────

def _parse_gpx(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        gpx = gpxpy.parse(f)

    total_distance = 0.0
    total_elevation = 0.0
    timestamps = []
    hr_values = []
    power_values = []
    cadence_values = []
    temp_values = []

    for track in gpx.tracks:
        for segment in track.segments:
            prev_point = None
            for point in segment.points:
                if prev_point:
                    total_distance += point.distance_2d(prev_point) or 0
                    ele_diff = (point.elevation or 0) - (prev_point.elevation or 0)
                    if ele_diff > 0:
                        total_elevation += ele_diff

                if point.time:
                    timestamps.append(point.time)

                # Extensions (HR, power, cadence from Garmin/Wahoo GPX)
                ext_data = _extract_gpx_extensions(point)
                if ext_data.get("hr"):
                    hr_values.append(ext_data["hr"])
                if ext_data.get("power"):
                    power_values.append(ext_data["power"])
                if ext_data.get("cadence"):
                    cadence_values.append(ext_data["cadence"])
                if ext_data.get("temp"):
                    temp_values.append(ext_data["temp"])

                prev_point = point

    duration = 0
    if len(timestamps) >= 2:
        duration = int((max(timestamps) - min(timestamps)).total_seconds())

    start_time = min(timestamps) if timestamps else None

    result: dict[str, Any] = {
        "name": gpx.tracks[0].name if gpx.tracks and gpx.tracks[0].name else "GPX Ride",
        "date": start_time.replace(tzinfo=timezone.utc) if start_time else None,
        "duration_seconds": duration,
        "distance_meters": total_distance,
        "elevation_gain_meters": total_elevation,
        "source": "gpx",
    }

    if hr_values:
        result["avg_hr"] = int(sum(hr_values) / len(hr_values))
        result["max_hr"] = int(max(hr_values))

    if power_values:
        result["avg_power"] = round(sum(power_values) / len(power_values), 1)
        result["max_power"] = round(max(power_values), 1)
        result["normalized_power"] = round(_calc_np(power_values), 1)

    if cadence_values:
        result["avg_cadence"] = int(sum(cadence_values) / len(cadence_values))

    if temp_values:
        result["temperature_c"] = round(sum(temp_values) / len(temp_values), 1)

    if power_values and hr_values and duration >= 2700:  # require 45+ min for drift
        drift, _ = compute_hr_drift_from_streams(power_values, hr_values)
        if drift is not None:
            result["hr_drift"] = drift

    return result


def _extract_gpx_extensions(element) -> dict:
    """
    Recursively extract HR, power, cadence, temp from GPX extension elements.
    Works with both gpxpy trackpoints (have .extensions list) and raw XML elements
    (iterable directly). Strips XML namespaces before matching tag names.
    """
    data = {}

    # Get the list of child elements: gpxpy points expose .extensions,
    # raw lxml/ET elements are directly iterable.
    if hasattr(element, "extensions") and element.extensions is not None:
        children = element.extensions
    else:
        children = list(element)

    for child in children:
        raw_tag = child.tag if hasattr(child, "tag") else ""
        # Strip namespace: {http://www.garmin.com/...}hr  →  hr
        tag = raw_tag.split("}")[-1].lower() if "}" in raw_tag else raw_tag.lower()

        text = (child.text or "").strip()

        if text:
            if tag in ("hr", "heartrate", "heartratebpm", "value"):
                # "value" appears inside <HeartRateBpm><Value>...</Value>
                # only accept if parent context is HR — guard with range check
                try:
                    v = int(float(text))
                    if 20 <= v <= 250:  # plausible HR range
                        data.setdefault("hr", v)
                except ValueError:
                    pass
            if tag == "power" or (tag.endswith("power") and "power" not in data):
                try: data["power"] = float(text)
                except ValueError: pass
            if tag in ("cadence", "cad"):
                try: data.setdefault("cadence", int(float(text)))
                except ValueError: pass
            if tag in ("atemp", "temp", "airtemp", "temperature"):
                try: data.setdefault("temp", float(text))
                except ValueError: pass

        # Recurse — this now works because we use list(child) for XML elements
        child_data = _extract_gpx_extensions(child)
        for k, v in child_data.items():
            data.setdefault(k, v)

    return data


# ─── FIT ────────────────────────────────────────────────────────────────────────

def _parse_fit(path: str) -> dict[str, Any]:
    fit = fitparse.FitFile(path)

    records = []
    session_data: dict[str, Any] = {}

    for message in fit.get_messages():
        if message.name == "record":
            rec = {d.name: d.value for d in message if d.value is not None}
            records.append(rec)
        elif message.name == "session":
            for d in message:
                if d.value is not None:
                    session_data[d.name] = d.value

    # Prefer session summary if available
    result: dict[str, Any] = {
        "name": "FIT Ride",
        "source": "fit",
    }

    if session_data.get("start_time"):
        t = session_data["start_time"]
        result["date"] = t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t

    result["duration_seconds"] = int(session_data.get("total_elapsed_time", 0))
    result["distance_meters"] = float(session_data.get("total_distance", 0))
    result["elevation_gain_meters"] = float(session_data.get("total_ascent", 0))
    result["avg_power"] = session_data.get("avg_power")
    result["max_power"] = session_data.get("max_power")
    result["normalized_power"] = session_data.get("normalized_power")
    result["avg_hr"] = session_data.get("avg_heart_rate")
    result["max_hr"] = session_data.get("max_heart_rate")
    result["avg_cadence"] = session_data.get("avg_cadence")

    # Fall back to computing from records if no session
    if records and not session_data.get("avg_power") and any("power" in r for r in records):
        powers = [r["power"] for r in records if "power" in r]
        result["avg_power"] = round(sum(powers) / len(powers), 1)
        result["max_power"] = round(max(powers), 1)
        result["normalized_power"] = round(_calc_np(powers), 1)

    temps = [r.get("temperature") for r in records if r.get("temperature") is not None]
    if temps:
        result["temperature_c"] = round(sum(temps) / len(temps), 1)

    # HR drift from per-record streams (requires both power + HR + 45+ min ride)
    fit_powers = [float(r["power"]) for r in records if "power" in r]
    fit_hrs    = [float(r["heart_rate"]) for r in records if "heart_rate" in r]
    if fit_powers and fit_hrs and result.get("duration_seconds", 0) >= 2700:
        drift, _ = compute_hr_drift_from_streams(fit_powers, fit_hrs)
        if drift is not None:
            result["hr_drift"] = drift

    return {k: v for k, v in result.items() if v is not None}


# ─── TCX ────────────────────────────────────────────────────────────────────────

def _parse_tcx(path: str) -> dict[str, Any]:
    from lxml import etree  # type: ignore

    tree = etree.parse(path)
    root = tree.getroot()
    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    def find(el, tag):
        return el.find(f"tcx:{tag}", ns)

    def findall(el, tag):
        return el.findall(f"tcx:{tag}", ns)

    activities = root.findall(".//tcx:Activity", ns)
    if not activities:
        return {"name": "TCX Ride", "source": "tcx", "duration_seconds": 0, "distance_meters": 0}

    activity = activities[0]
    laps = findall(activity, "Lap")

    total_time = sum(float(find(lap, "TotalTimeSeconds").text) for lap in laps if find(lap, "TotalTimeSeconds") is not None)
    total_dist = sum(float(find(lap, "DistanceMeters").text) for lap in laps if find(lap, "DistanceMeters") is not None)
    total_elev = sum(float(find(lap, "TotalAscent").text) for lap in laps if find(lap, "TotalAscent") is not None)

    hr_values = [
        int(el.text)
        for lap in laps
        for el in lap.findall(".//tcx:HeartRateBpm/tcx:Value", ns)
    ]
    power_values = [
        float(el.text)
        for lap in laps
        for el in lap.findall(".//tcx:Watts", ns)
    ]

    start_el = activity.find(".//tcx:Id", ns)
    start_time = None
    if start_el is not None and start_el.text:
        from datetime import datetime
        start_time = datetime.fromisoformat(start_el.text.replace("Z", "+00:00"))

    result: dict[str, Any] = {
        "name": "TCX Ride",
        "source": "tcx",
        "duration_seconds": int(total_time),
        "distance_meters": total_dist,
        "elevation_gain_meters": total_elev if total_elev else None,
        "date": start_time,
    }

    if hr_values:
        result["avg_hr"] = int(sum(hr_values) / len(hr_values))
        result["max_hr"] = max(hr_values)

    if power_values:
        result["avg_power"] = round(sum(power_values) / len(power_values), 1)
        result["max_power"] = round(max(power_values), 1)
        result["normalized_power"] = round(_calc_np(power_values), 1)

    return result


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _calc_np(powers: list[float], sample_interval_s: int = 1) -> float:
    """
    Normalized Power (Coggan):
    1. 30-second rolling average
    2. Raise to 4th power
    3. Average
    4. Take 4th root
    """
    if not powers:
        return 0.0
    window = max(1, 30 // sample_interval_s)
    rolling = []
    for i in range(len(powers)):
        start = max(0, i - window + 1)
        chunk = powers[start : i + 1]
        rolling.append(sum(chunk) / len(chunk))

    avg_fourth = sum(v ** 4 for v in rolling) / len(rolling)
    return avg_fourth ** 0.25
