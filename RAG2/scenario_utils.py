"""
RAG2/scenario_utils.py

Null-friendly scenario utilities:
- Family detection
- Optional (soft) validation that NEVER raises for missing user_config fields
- Runtime hints attachment (non-persistent) for generators/LLM

Design goal (per your final requirement):
- Parser output may contain nulls.
- Nulls must NOT be treated as missing mandatory inputs.
- Codegen must not crash because validation hard-failed.

------------------------------------------------------------
Responsible for: Providing null-tolerant scenario helpers -- ADAS family
detection, AEB variant inference, and soft (never-raising) validation that
attaches non-persistent runtime_hints/warnings for generators to consult,
instead of hard-failing on missing user_config fields.
Maintainer: shamanth.adiga@ltts.com
------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Dict, List
import re

# -----------------------------------------------------------------------------
# Safe getters
# -----------------------------------------------------------------------------

# Returns a scenario's "user_config" block, or an empty dict if missing/not a dict.
def _get_uc(s: Dict[str, Any]) -> Dict[str, Any]:
    return (s.get("user_config") or {}) if isinstance(s, dict) else {}

# Returns a scenario's "scenario_details" block, or an empty dict if missing/not a dict.
def _get_sd(s: Dict[str, Any]) -> Dict[str, Any]:
    return (s.get("scenario_details") or {}) if isinstance(s, dict) else {}

# Returns a scenario's scenario_details.extra block, or an empty dict if missing.
def _get_extra(s: Dict[str, Any]) -> Dict[str, Any]:
    sd = _get_sd(s)
    return (sd.get("extra") or {}) if isinstance(sd, dict) else {}

# Returns a scenario's "classification" block, or an empty dict if missing/not a dict.
def _get_classification(s: Dict[str, Any]) -> Dict[str, Any]:
    return (s.get("classification") or {}) if isinstance(s, dict) else {}

# Walks a dot-separated key path through nested dicts, returning `default`
# if any step is missing or the final value is None.
def _get_path(obj: Any, path: str, default: Any = None) -> Any:
    """
    Safe dot-path getter.
    Returns `default` if the path doesn't exist or if the final value is None.
    """
    cur: Any = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur

# Treats both None and empty string as "empty" for soft-validation purposes.
def _is_empty(v: Any) -> bool:
    return v is None or v == ""

# -----------------------------------------------------------------------------
# Family detection
# -----------------------------------------------------------------------------

# Determines the scenario's ADAS family by checking, in order: the
# extra.adas_family tag, classification.family, then scenario-name keyword
# heuristics, defaulting to "AEB" if nothing matches.
def _family_of(s: Dict[str, Any]) -> str:
    """
    Determine scenario family: AEB / LSS / VRU.

    Priority:
      1) scenario_details.extra.adas_family
      2) classification.family
      3) scenario_name heuristics
      4) default AEB
    """
    extra = _get_extra(s)
    fam = str(extra.get("adas_family") or "").strip().upper()
    if fam in ("AEB", "LSS", "VRU"):
        return fam

    cls = _get_classification(s)
    fam2 = str(cls.get("family") or "").strip().upper()
    if fam2 in ("AEB", "LSS", "VRU"):
        return fam2

    name = str(s.get("scenario_name") or "").upper()
    if "LSS" in name or "LANE" in name:
        return "LSS"
    if "VRU" in name or "PED" in name or "BICY" in name:
        return "VRU"
    return "AEB"

# -----------------------------------------------------------------------------
# AEB variant hints (soft; do NOT enforce)
# -----------------------------------------------------------------------------

_AEB_VARIANTS = [
    ("ccrs", re.compile(r"\bCCRs\b", re.IGNORECASE)),
    ("ccrm", re.compile(r"\bCCRm\b", re.IGNORECASE)),
    ("ccrb", re.compile(r"\bCCRb\b", re.IGNORECASE)),
    ("ccf_tap", re.compile(r"\bCCFtap\b", re.IGNORECASE)),
    ("ccf_hol", re.compile(r"\bCCFhol\b", re.IGNORECASE)),
    ("cccscp", re.compile(r"\bCCCscp\b", re.IGNORECASE)),
]

# Soft-infers the AEB scenario variant (ccrs/ccrm/ccrb/etc.) from
# classification.variant first, falling back to a regex match against the
# scenario name, or "unknown" if neither yields a match.
def _infer_aeb_variant(s: Dict[str, Any]) -> str:
    """
    Soft inference for AEB scenario variant.
    Uses classification.variant first; then pattern match on scenario_name.
    """
    cls = _get_classification(s)
    v = str(cls.get("variant") or "").strip().lower()
    if v:
        return v

    name = str(s.get("scenario_name") or "")
    for key, rx in _AEB_VARIANTS:
        if rx.search(name):
            return key

    return "unknown"

# Appends a non-blocking warning message to the scenario's
# runtime_hints.warnings list (in place), creating that structure if it doesn't exist yet.
def _add_warning(s: Dict[str, Any], msg: str) -> None:
    """
    Attach warnings under runtime_hints.warnings (non-persistent).
    Warnings never block generation.
    """
    if not isinstance(s, dict):
        return
    hints = s.setdefault("runtime_hints", {})
    if not isinstance(hints, dict):
        s["runtime_hints"] = {}
        hints = s["runtime_hints"]
    warnings = hints.setdefault("warnings", [])
    if isinstance(warnings, list):
        warnings.append(msg)

# Attaches non-persistent runtime_hints to the scenario (currently just
# the inferred AEB variant for AEB-family scenarios) without altering the canonical schema.
def _attach_runtime_hints(s: Dict[str, Any]) -> None:
    """
    Attach non-persistent runtime hints for generators/LLM.
    Does NOT modify the canonical schema or enforce anything.
    """
    if not isinstance(s, dict):
        return

    fam = _family_of(s)
    hints = s.setdefault("runtime_hints", {})
    if not isinstance(hints, dict):
        s["runtime_hints"] = {}
        hints = s["runtime_hints"]

    if fam == "AEB":
        hints["aeb_variant"] = _infer_aeb_variant(s)

# -----------------------------------------------------------------------------
# Soft validation (NEVER raises for missing fields)
# -----------------------------------------------------------------------------

# Walks every user_config field that generators care about (map/entities/
# termination/trigger/dynamics/layout/behavior/VRU motion) and attaches a
# soft warning for each one that's null or only partially specified,
# without ever raising except when the scenario itself isn't a dict.
def _ensure_mandatory_user_config(s: Dict[str, Any]) -> None:
    """
    Historically this function hard-failed scenarios missing certain user_config
    fields. That behavior is NOT compatible with a parser-first pipeline where
    fields may legitimately be null.

    New behavior:
    - Never raises due to missing/null user_config fields.
    - Only raises if the scenario object itself is invalid (not a dict).
    - Attaches runtime_hints + warnings to help generators/LLM choose defaults.
    """
    if not isinstance(s, dict):
        raise RuntimeError("Scenario must be a dict.")

    fam = _family_of(s)

    # Always attach runtime hints (non-blocking)
    _attach_runtime_hints(s)

    # Soft checks: warn only. Do NOT require.
    # These are *useful* to know for deterministic placement/behavior.
    town = _get_path(s, "user_config.map.town", None)
    if _is_empty(town):
        _add_warning(s, "user_config.map.town is null; generator should pick a default town/map.")

    ego_bp = _get_path(s, "user_config.entities.ego.blueprint", None)
    if _is_empty(ego_bp):
        _add_warning(s, "user_config.entities.ego.blueprint is null; generator should default ego vehicle blueprint.")

    if fam == "AEB":
        tgt_bp = _get_path(s, "user_config.entities.target.blueprint", None)
        if _is_empty(tgt_bp):
            _add_warning(s, "AEB: user_config.entities.target.blueprint is null; generator should default target vehicle blueprint.")

    timeout_s = _get_path(s, "user_config.termination.timeout_s", None)
    if _is_empty(timeout_s):
        _add_warning(s, "user_config.termination.timeout_s is null; generator should default timeout (e.g., 60s).")

    # Trigger: allow START_IMMEDIATELY with both params null.
    trig_type = str(_get_path(s, "user_config.trigger.type", "")).strip().upper()
    if _is_empty(trig_type):
        _add_warning(s, "user_config.trigger.type is null; generator should default to START_IMMEDIATELY.")
    else:
        if trig_type == "DISTANCE":
            if _is_empty(_get_path(s, "user_config.trigger.distance_m", None)):
                _add_warning(s, "Trigger type DISTANCE but distance_m is null; generator should default distance or skip distance-based trigger.")
        elif trig_type == "TTC":
            if _is_empty(_get_path(s, "user_config.trigger.ttc_s", None)):
                _add_warning(s, "Trigger type TTC but ttc_s is null; generator should default TTC or approximate TTC trigger.")
        # START_IMMEDIATELY -> no warning needed.

    # Speeds: allow null. Warn
        # Speeds: allow null. Warn
    ego_kph = _get_path(s, "user_config.dynamics.ego_speed_kph", None)
    ego_kph_min = _get_path(s, "user_config.dynamics.ego_speed_kph_min", None)
    if _is_empty(ego_kph) and _is_empty(ego_kph_min):
        _add_warning(s, "user_config.dynamics.ego_speed_kph and ego_speed_kph_min are null; generator should default ego speed (e.g., 30–50 kph).")

    if fam == "AEB":
        tgt_kph = _get_path(s, "user_config.dynamics.target_speed_kph", None)
        tgt_kph_min = _get_path(s, "user_config.dynamics.target_speed_kph_min", None)
        if _is_empty(tgt_kph) and _is_empty(tgt_kph_min):
            _add_warning(s, "AEB: user_config.dynamics.target_speed_kph and target_speed_kph_min are null; generator should default target speed based on variant (e.g., 0 for stationary).")

    # Layout: allow null. Warn (gap/distance helps initial placement)
    gap = _get_path(s, "user_config.layout.initial_gap_m", None)
    dist = _get_path(s, "user_config.layout.initial_distance_m", None)
    if _is_empty(gap) and _is_empty(dist):
        _add_warning(s, "user_config.layout.initial_gap_m and initial_distance_m are null; generator should default gap/distance (e.g., 15m).")

    lat = _get_path(s, "user_config.layout.lateral_offset_m", None)
    if _is_empty(lat):
        _add_warning(s, "user_config.layout.lateral_offset_m is null; generator should default lateral offset to 0.0m.")

    # overlap_percent is intentionally OPTIONAL and may be absent/null
    # If present, generator may convert it to lateral offset; if null, ignore.
    ov = _get_path(s, "user_config.layout.overlap_percent", None)
    if ov is not None and isinstance(ov, (int, float)):
        # no warning; just leave as hint-friendly field
        pass

    # Behavior: allow null. Warn only if fields are partially specified
    target_behavior = _get_path(s, "user_config.behavior.target_behavior", None)
    if fam == "AEB" and _is_empty(target_behavior):
        _add_warning(s, "AEB: user_config.behavior.target_behavior is null; generator should default to 'constant_speed' unless variant implies braking/cut-in.")

    # Lane-change fields are optional; warn only if a maneuver is partially defined
    lc_type = _get_path(s, "user_config.behavior.lane_change.maneuver_type", None)
    lc_dir = _get_path(s, "user_config.behavior.lane_change.direction", None)
    lc_dur = _get_path(s, "user_config.behavior.lane_change.duration_s", None)
    lc_dist = _get_path(s, "user_config.behavior.lane_change.distance_m", None)
    if not _is_empty(lc_type):
        # They asked for lane-change but left all parameters empty
        if _is_empty(lc_dir) and _is_empty(lc_dur) and _is_empty(lc_dist):
            _add_warning(s, "lane_change.maneuver_type is set but direction/duration/distance are null; generator should pick safe defaults for lane change.")

    # Braking fields optional; warn if decel specified without duration (or vice versa)
    tgt_decel = _get_path(s, "user_config.behavior.target_decel_mps2", None)
    tgt_brk_dur = _get_path(s, "user_config.behavior.target_brake_duration_s", None)
    if (tgt_decel is not None) ^ (tgt_brk_dur is not None):
        _add_warning(s, "target braking is partially specified (decel xor duration); generator should infer missing value or ignore braking action.")

    ego_decel = _get_path(s, "user_config.behavior.ego_decel_mps2", None)
    if ego_decel is not None and not isinstance(ego_decel, (int, float)):
        _add_warning(s, "ego_decel_mps2 is non-numeric; generator should ignore or coerce safely.")

    # VRU: allow null; if vru blueprint exists but motion absent, warn
    vru_bp = _get_path(s, "user_config.entities.vru.blueprint", None)
    if fam == "VRU" and not _is_empty(vru_bp):
        vru_speed = _get_path(s, "user_config.behavior.vru_motion.speed_mps", None)
        vru_side = _get_path(s, "user_config.behavior.vru_motion.crossing_side", None)
        if _is_empty(vru_speed) and _is_empty(vru_side):
            _add_warning(s, "VRU: vru blueprint present but vru_motion is empty; generator should default crossing speed/side.")

    # No return needed; scenario mutated in-place with runtime_hints only.


# -----------------------------------------------------------------------------
# Optional normalization helpers (safe, minimal)
# -----------------------------------------------------------------------------

# Public wrapper around _ensure_mandatory_user_config that guarantees no
# exception ever escapes -- any internal error is itself recorded as a
# warning on the scenario rather than propagated.
def ensure_runtime_hints(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public helper: attach runtime_hints and soft warnings, without raising.
    """
    try:
        _ensure_mandatory_user_config(s)
    except Exception as e:
        # Even here: avoid hard failure for pipeline robustness
        if isinstance(s, dict):
            _add_warning(s, f"ensure_runtime_hints encountered error: {e!r}")
    return s


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

__all__ = [
    "_family_of",
    "_ensure_mandatory_user_config",
    "ensure_runtime_hints",
    "_get_path",
]
