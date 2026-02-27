"""PARSER/main_parser.py

Purpose
-------
Run the full 3-stage Euro NCAP PDF parsing pipeline and return scenarios to the UI.

**Output contract (IMPORTANT):**
This module returns a *uniform scenario schema* for all ADAS families:
- AEB
- LSS
- VRU

This avoids schema confusion caused by overlapping/duplicated fields (e.g. town,
blueprints) across top-level keys and user_config.

Pipeline stages
---------------
Stage 1:
  parser_1 + pdf_to_json_raw  -> knowledge_base_raw.json

Stage 2:
  stage2_build_structured_and_evidence ->
    structured_scenarios.json
    scenario_evidence.json

Stage 3 (optional):
  llm_enricher.enrich_all -> enriched_scenarios.json

Notes
-----
- This module does NOT call the Scenario generation LLM.
- It prepares a clean scenario snippet that downstream generators can consume.
"""

from __future__ import annotations

import json
import os
import inspect
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import easyocr  # optional; only used if parser_1.process_pdf requires a reader
except Exception:  # pragma: no cover
    easyocr = None

from PARSER import parser_1
from PARSER import pdf_to_json_raw
from PARSER import stage2_build_structured_and_evidence
from PARSER.llm_enricher import enrich_all


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_rmtree(path: str, retries: int = 8, delay: float = 0.35) -> bool:
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            time.sleep(delay)
        except FileNotFoundError:
            return True
    shutil.rmtree(path, ignore_errors=True)
    return True


def _project_root() -> Path:
    """
    Resolve project root robustly, regardless of Streamlit CWD.

    Expected layout:
      <ROOT>/
        UI/
        PARSER/
        RAG2/
        Parsed_Data/ (optional)
    """
    here = Path(__file__).resolve()
    # .../PARSER/main_parser.py -> parent is PARSER, parent.parent is root
    return here.parent.parent


PROJECT_ROOT = _project_root()
BASE_PDF_FOLDER = PROJECT_ROOT / "EuroNcap"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "Parsed_Data"


# -----------------------------------------------------------------------------
# Family / subtype normalization
# -----------------------------------------------------------------------------

_LSS_HINTS = ("ELK", "LKA", "LDW", "BSM", "LSS")
_VRU_HINTS = (
    "VRU",
    "PEDESTRIAN",
    "CYCLIST",
    "BICYCLIST",
    "MOTORCYCLIST",
    "DOORING",  # many VRU dooring titles include this
    # common Euro NCAP codes for VRU:
    "EPTA",
    "EPTC",
    "EBTA",
    "EMT",
    "CPNA",
    "CPFA",
    "CPLA",
    "CPNCO",
    "CPTA",
    "CPRA",
    "CBNA",
    "CBFA",
    "CBLA",
    "CBTA",
    "CBDA",
    "CMRS",
    "CMRB",
    "CMFTAP",
    "CMONCOMING",
    "CMOVERTAKING",
)


def _safe_upper(v: Any) -> str:
    return str(v or "").strip().upper()


def _parse_family_and_subtype(
    raw: Any, scenario_code: str = "", scenario_name: str = ""
) -> Tuple[str, Optional[str]]:
    """Normalize family labels into one of {AEB, LSS, VRU} and optionally a subtype.

    Examples:
      - "AEB_CCR" or "AEB_CCRS" -> ("AEB", "CCR")
      - "AEB" -> ("AEB", None)
      - "LSS" -> ("LSS", None)
      - "VRU" -> ("VRU", None)
      - Unknown -> inferred heuristically (defaults to AEB)
    """

    raw_u = _safe_upper(raw)

    # Direct families
    if raw_u in {"AEB", "LSS", "VRU"}:
        return raw_u, None

    # Prefix style e.g. AEB_CCR
    if raw_u.startswith("AEB"):
        parts = raw_u.split("_")
        if len(parts) >= 2:
            suf = parts[1]
            if suf.startswith("CCR"):
                return "AEB", "CCR"
            if suf.startswith("CCF"):
                return "AEB", "CCF"
            if suf.startswith("CCB"):
                return "AEB", "CCB"
        return "AEB", None

    if raw_u.startswith("LSS"):
        return "LSS", None

    if raw_u.startswith("VRU"):
        return "VRU", None

    # Heuristics based on scenario_code/name
    code_u = _safe_upper(scenario_code)
    name_u = _safe_upper(scenario_name)

    # VRU pattern hints
    if code_u.startswith("VRU_") or code_u.startswith(("CP", "CB", "CM")):
        return "VRU", None
    if any(k in name_u for k in _VRU_HINTS):
        return "VRU", None

    # LSS pattern hints
    if code_u.startswith("LSS_"):
        return "LSS", None
    if any(k in name_u for k in _LSS_HINTS):
        return "LSS", None

    # AEB common patterns
    if code_u.startswith(("CCR", "CCF", "CCB", "CCC", "CCRM", "CCRS")):
        if code_u.startswith("CCR"):
            return "AEB", "CCR"
        if code_u.startswith("CCF"):
            return "AEB", "CCF"
        if code_u.startswith("CCB"):
            return "AEB", "CCB"
        if code_u.startswith("CCC"):
            return "AEB", "CCC"
        return "AEB", None

    # Default fallback
    return "AEB", None


def _ensure_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _get_nested(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return default if cur is None else cur


def _tag_adas_family(s: Dict[str, Any]) -> None:
    """Attach normalized adas family tags to scenario_details.extra."""
    if not isinstance(s, dict):
        return

    sd = _ensure_dict(s.get("scenario_details"))
    s["scenario_details"] = sd

    extra = _ensure_dict(sd.get("extra"))
    sd["extra"] = extra

    scenario_name = str(s.get("scenario_name") or sd.get("scenario") or "")
    scenario_code = str(s.get("scenario_code") or "")

    fam, subtype = _parse_family_and_subtype(
        extra.get("adas_family"), scenario_code, scenario_name
    )

    extra["adas_family"] = fam
    if subtype:
        extra["adas_subtype"] = subtype
    else:
        extra.pop("adas_subtype", None)


def _detect_variant(scenario_name: str, scenario_code: str, family: str) -> Optional[str]:
    """Detect special variants to pre-create extension blocks (null placeholders)."""
    name_u = _safe_upper(scenario_name)
    code_u = _safe_upper(scenario_code)

    if family == "AEB":
        if "STATIONARY" in name_u or code_u.endswith("S"):
            return "rear_stationary"
        if "MOVING" in name_u or code_u.endswith("M"):
            return "rear_moving"
        return None

    if family == "LSS":
        if "CUT" in name_u and "IN" in name_u:
            return "cut_in"
        if "CUT" in name_u and "OUT" in name_u:
            return "cut_out"
        return None

    if family == "VRU":
        if "DOOR" in name_u or "DOORING" in name_u or code_u == "CBDA":
            return "dooring"
        if "CROSS" in name_u:
            return "crossing"
        return None

    return None


def _default_extensions_for_variant(variant: Optional[str]) -> Dict[str, Any]:
    """Return extension blocks with null placeholders when a special variant is detected."""
    if variant == "dooring":
        return {
            "door": {
                "door_side": None,  # left/right
                "door_open_time_s": None,
                "door_open_angle_deg": None,
                "door_actor_blueprint": None,
                "door_pivot_offset_m": None,
            }
        }
    return {}


# -----------------------------------------------------------------------------
# SANITIZE: hard schema enforcement (prevents ALL legacy / duplicate keys)
# -----------------------------------------------------------------------------

_ALLOWED_TOP_LEVEL = {
    "id",
    "domain",
    "name",
    "classification",
    "source",
    "scenario_details",
    "user_config",
}

_ALLOWED_USER_CONFIG = {
    "map",
    "entities",
    "layout",
    "trigger",
    "dynamics",
    "behavior",
    "termination",
    "extensions",
}

_ALLOWED_MAP = {"town", "road_selection_mode", "road_id", "lane_id"}
_ALLOWED_ENTITIES = {"ego", "target", "vru"}
_ALLOWED_ENTITY = {"blueprint", "spawn"}
_ALLOWED_LAYOUT = {"initial_gap_m", "initial_distance_m", "lateral_offset_m", "overlap_percent"}
_ALLOWED_TRIGGER = {"type", "distance_m", "ttc_s"}
_ALLOWED_DYNAMICS = {
    "ego_speed_kph",
    "ego_speed_kph_min",
    "ego_speed_kph_max",
    "target_speed_kph",
    "target_speed_kph_min",
    "target_speed_kph_max",
}
_ALLOWED_BEHAVIOR = {
    "scenario_variant",
    "target_behavior",
    "target_decel_mps2",
    "target_brake_duration_s",
    "ego_decel_mps2",
    "lane_change",
    "vru_motion",
}
_ALLOWED_LANE_CHANGE = {"direction", "duration_s", "distance_m", "lateral_offset_m", "maneuver_type"}
_ALLOWED_VRU_MOTION = {"speed_mps_min", "speed_mps_max", "speed_mps", "crossing_side", "path_offset_m"}
_ALLOWED_TERMINATION = {"timeout_s", "stop_on_collision"}
_ALLOWED_EXTENSIONS = {"notes", "door"}  # door is optional variant extension


def _dict_keep_only(d: Dict[str, Any], allowed: set) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    return {k: d.get(k) for k in allowed if k in d}


def _sanitize_uniform_scenario(out: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce ONLY the canonical uniform schema and delete everything else."""
    if not isinstance(out, dict):
        return {}

    cleaned = _dict_keep_only(out, _ALLOWED_TOP_LEVEL)

    if not isinstance(cleaned.get("classification"), dict):
        cleaned["classification"] = {"family": None, "subtype": None, "variant": None}
    if not isinstance(cleaned.get("source"), dict):
        cleaned["source"] = {"standard": "Euro NCAP", "evidence": []}
    if not isinstance(cleaned.get("scenario_details"), dict):
        cleaned["scenario_details"] = {}
    if not isinstance(cleaned.get("user_config"), dict):
        cleaned["user_config"] = {}

    uc = cleaned["user_config"]
    uc = _dict_keep_only(uc, _ALLOWED_USER_CONFIG)

    uc_map = _ensure_dict(uc.get("map"))
    uc_entities = _ensure_dict(uc.get("entities"))
    uc_layout = _ensure_dict(uc.get("layout"))
    uc_trigger = _ensure_dict(uc.get("trigger"))
    uc_dynamics = _ensure_dict(uc.get("dynamics"))
    uc_behavior = _ensure_dict(uc.get("behavior"))
    uc_term = _ensure_dict(uc.get("termination"))
    uc_ext = _ensure_dict(uc.get("extensions"))

    uc["map"] = _dict_keep_only(uc_map, _ALLOWED_MAP)

    ent_clean: Dict[str, Any] = {}
    for ent_key in _ALLOWED_ENTITIES:
        ent_val = _ensure_dict(uc_entities.get(ent_key))
        ent_clean[ent_key] = _dict_keep_only(ent_val, _ALLOWED_ENTITY)
    uc["entities"] = ent_clean

    uc["layout"] = _dict_keep_only(uc_layout, _ALLOWED_LAYOUT)
    uc["trigger"] = _dict_keep_only(uc_trigger, _ALLOWED_TRIGGER)
    uc["dynamics"] = _dict_keep_only(uc_dynamics, _ALLOWED_DYNAMICS)

    beh_clean = _dict_keep_only(uc_behavior, _ALLOWED_BEHAVIOR)
    lane_change = _ensure_dict(uc_behavior.get("lane_change"))
    vru_motion = _ensure_dict(uc_behavior.get("vru_motion"))
    beh_clean["lane_change"] = _dict_keep_only(lane_change, _ALLOWED_LANE_CHANGE)
    beh_clean["vru_motion"] = _dict_keep_only(vru_motion, _ALLOWED_VRU_MOTION)
    uc["behavior"] = beh_clean

    uc["termination"] = _dict_keep_only(uc_term, _ALLOWED_TERMINATION)

    ext_clean = _dict_keep_only(uc_ext, _ALLOWED_EXTENSIONS)
    uc["extensions"] = ext_clean

    cleaned["user_config"] = uc
    return cleaned


def _build_uniform_schema(master: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a parsed scenario dict into the uniform schema."""
    sd = _ensure_dict(master.get("scenario_details"))
    extra = _ensure_dict(sd.get("extra"))

    scenario_name = str(master.get("scenario_name") or sd.get("scenario") or "")
    scenario_code = str(master.get("scenario_code") or "")

    fam, subtype = _parse_family_and_subtype(extra.get("adas_family"), scenario_code, scenario_name)

    uc_in = _ensure_dict(master.get("user_config"))

    town = _get_nested(uc_in, ["map", "town"], None) or uc_in.get("town") or master.get("town")
    road_sel_mode = _get_nested(uc_in, ["map", "road_selection_mode"], None) or uc_in.get("road_selection_mode")

    ego_bp = _get_nested(uc_in, ["entities", "ego", "blueprint"], None) or uc_in.get("ego_vehicle_blueprint") or master.get("vehicle_blueprint")
    tgt_bp = _get_nested(uc_in, ["entities", "target", "blueprint"], None) or uc_in.get("target_vehicle_blueprint") or master.get("target_vehicle_blueprint")

    ego_spawn = _get_nested(uc_in, ["entities", "ego", "spawn"], None) or _get_nested(uc_in, ["spawn", "ego"], None)
    tgt_spawn = _get_nested(uc_in, ["entities", "target", "spawn"], None) or _get_nested(uc_in, ["spawn", "target"], None)
    vru_spawn = _get_nested(uc_in, ["entities", "vru", "spawn"], None) or _get_nested(uc_in, ["spawn", "vru"], None)

    initial_gap_m = _get_nested(uc_in, ["layout", "initial_gap_m"], None) or uc_in.get("initial_gap_m")
    initial_distance_m = _get_nested(uc_in, ["layout", "initial_distance_m"], None) or uc_in.get("initial_distance_m")
    lateral_offset_m = _get_nested(uc_in, ["layout", "lateral_offset_m"], None) or uc_in.get("lateral_offset_m")
    overlap_percent = _get_nested(uc_in, ["layout", "overlap_percent"], None) or uc_in.get("overlap_percent")

    trig_type = _get_nested(uc_in, ["trigger", "type"], None)
    trig_distance = _get_nested(uc_in, ["trigger", "distance_m"], None) or _get_nested(uc_in, ["trigger", "distance_m"], None)
    trig_ttc = _get_nested(uc_in, ["trigger", "ttc_s"], None)

    timeout_s = _get_nested(uc_in, ["termination", "timeout_s"], None) or uc_in.get("timeout_s")
    stop_on_collision = _get_nested(uc_in, ["termination", "stop_on_collision"], None) or uc_in.get("stop_on_collision")

    ego_speed_kph = _get_nested(uc_in, ["dynamics", "ego_speed_kph"], None) or _get_nested(uc_in, ["aeb", "ego_speed_kph"], None) or uc_in.get("ego_speed_kph")
    ego_speed_kph_min = _get_nested(uc_in, ["dynamics", "ego_speed_kph_min"], None) or uc_in.get("ego_speed_kph_min") or _get_nested(uc_in, ["aeb", "ego_speed_kph_min"], None) or _get_nested(uc_in, ["lss", "ego_speed_kph_min"], None)
    ego_speed_kph_max = _get_nested(uc_in, ["dynamics", "ego_speed_kph_max"], None) or uc_in.get("ego_speed_kph_max") or _get_nested(uc_in, ["aeb", "ego_speed_kph_max"], None) or _get_nested(uc_in, ["lss", "ego_speed_kph_max"], None)

    target_speed_kph = _get_nested(uc_in, ["dynamics", "target_speed_kph"], None) or _get_nested(uc_in, ["aeb", "target_speed_kph"], None) or uc_in.get("target_speed_kph")
    target_speed_kph_min = _get_nested(uc_in, ["dynamics", "target_speed_kph_min"], None) or _get_nested(uc_in, ["aeb", "target_speed_kph_min"], None)
    target_speed_kph_max = _get_nested(uc_in, ["dynamics", "target_speed_kph_max"], None) or _get_nested(uc_in, ["aeb", "target_speed_kph_max"], None)

    scenario_variant = _get_nested(uc_in, ["behavior", "scenario_variant"], None) or _get_nested(uc_in, ["aeb", "scenario_variant"], None) or _get_nested(uc_in, ["lss", "maneuver_type"], None) or _get_nested(uc_in, ["vru", "scenario_variant"], None)
    target_behavior = _get_nested(uc_in, ["behavior", "target_behavior"], None) or _get_nested(uc_in, ["aeb", "target_behavior"], None)
    target_decel_mps2 = _get_nested(uc_in, ["behavior", "target_decel_mps2"], None) or _get_nested(uc_in, ["aeb", "target_decel_mps2"], None)
    target_brake_duration_s = _get_nested(uc_in, ["behavior", "target_brake_duration_s"], None) or _get_nested(uc_in, ["aeb", "target_brake_duration_s"], None)
    ego_decel_mps2 = _get_nested(uc_in, ["behavior", "ego_decel_mps2"], None) or _get_nested(uc_in, ["aeb", "ego_decel_mps2"], None)

    lane_change_direction = _get_nested(uc_in, ["behavior", "lane_change", "direction"], None) or _get_nested(uc_in, ["lss", "lane_change_direction"], None)
    lane_change_duration_s = _get_nested(uc_in, ["behavior", "lane_change", "duration_s"], None) or _get_nested(uc_in, ["lss", "lane_change_duration_s"], None)
    lane_change_distance_m = _get_nested(uc_in, ["behavior", "lane_change", "distance_m"], None) or _get_nested(uc_in, ["lss", "lane_change_distance_m"], None)
    lane_change_lateral_offset_m = _get_nested(uc_in, ["behavior", "lane_change", "lateral_offset_m"], None) or _get_nested(uc_in, ["lss", "lateral_offset_m"], None)
    lane_change_maneuver_type = _get_nested(uc_in, ["behavior", "lane_change", "maneuver_type"], None) or _get_nested(uc_in, ["lss", "maneuver_type"], None)

    vru_actor_blueprint = _get_nested(uc_in, ["entities", "vru", "blueprint"], None) or _get_nested(uc_in, ["vru", "actor_blueprint"], None)
    vru_speed_mps_min = _get_nested(uc_in, ["behavior", "vru_motion", "speed_mps_min"], None) or _get_nested(uc_in, ["vru", "vru_speed_mps_min"], None)
    vru_speed_mps_max = _get_nested(uc_in, ["behavior", "vru_motion", "speed_mps_max"], None) or _get_nested(uc_in, ["vru", "vru_speed_mps_max"], None)
    vru_speed_mps = _get_nested(uc_in, ["behavior", "vru_motion", "speed_mps"], None) or _get_nested(uc_in, ["vru", "vru_speed_mps"], None)
    vru_crossing_side = _get_nested(uc_in, ["behavior", "vru_motion", "crossing_side"], None) or _get_nested(uc_in, ["vru", "crossing_side"], None)
    vru_path_offset_m = _get_nested(uc_in, ["behavior", "vru_motion", "path_offset_m"], None) or _get_nested(uc_in, ["vru", "path_offset_m"], None)

    notes_ext = _get_nested(uc_in, ["extensions", "notes"], None) or _get_nested(uc_in, ["user_config", "extensions", "notes"], None)

    out: Dict[str, Any] = {
        "id": master.get("id") or master.get("scenario_id") or None,
        "domain": "ADAS",
        "name": scenario_name,
        "classification": {"family": fam, "subtype": subtype, "variant": None},
        "source": {
            "standard": extra.get("standard") or "Euro NCAP",
            "evidence": extra.get("evidence") or [],
        },
        "scenario_details": sd,
        "user_config": {
            "map": {
                "town": town,
                "road_selection_mode": road_sel_mode,
                "road_id": _get_nested(uc_in, ["map", "road_id"], None),
                "lane_id": _get_nested(uc_in, ["map", "lane_id"], None),
            },
            "entities": {
                "ego": {"blueprint": ego_bp, "spawn": ego_spawn},
                "target": {"blueprint": tgt_bp, "spawn": tgt_spawn},
                "vru": {"blueprint": vru_actor_blueprint, "spawn": vru_spawn},
            },
            "layout": {
                "initial_gap_m": initial_gap_m,
                "initial_distance_m": initial_distance_m,
                "lateral_offset_m": lateral_offset_m,
                "overlap_percent": overlap_percent,
            },
            "trigger": {"type": trig_type, "distance_m": trig_distance, "ttc_s": trig_ttc},
            "dynamics": {
                "ego_speed_kph": ego_speed_kph,
                "ego_speed_kph_min": ego_speed_kph_min,
                "ego_speed_kph_max": ego_speed_kph_max,
                "target_speed_kph": target_speed_kph,
                "target_speed_kph_min": target_speed_kph_min,
                "target_speed_kph_max": target_speed_kph_max,
            },
            "behavior": {
                "scenario_variant": scenario_variant,
                "target_behavior": target_behavior,
                "target_decel_mps2": target_decel_mps2,
                "target_brake_duration_s": target_brake_duration_s,
                "ego_decel_mps2": ego_decel_mps2,
                "lane_change": {
                    "direction": lane_change_direction,
                    "duration_s": lane_change_duration_s,
                    "distance_m": lane_change_distance_m,
                    "lateral_offset_m": lane_change_lateral_offset_m,
                    "maneuver_type": lane_change_maneuver_type,
                },
                "vru_motion": {
                    "speed_mps_min": vru_speed_mps_min,
                    "speed_mps_max": vru_speed_mps_max,
                    "speed_mps": vru_speed_mps,
                    "crossing_side": vru_crossing_side,
                    "path_offset_m": vru_path_offset_m,
                },
            },
            "termination": {"timeout_s": timeout_s, "stop_on_collision": stop_on_collision},
            "extensions": {"notes": notes_ext},
        },
    }

    variant = _detect_variant(scenario_name, scenario_code, fam)
    if variant:
        out["classification"]["variant"] = variant
        ext = _default_extensions_for_variant(variant)
        out["user_config"]["extensions"].update(ext)

    sd_out = out.get("scenario_details")
    if isinstance(sd_out, dict):
        extra_out = sd_out.setdefault("extra", {})
        if isinstance(extra_out, dict):
            extra_out["adas_family"] = fam
            if subtype:
                extra_out["adas_subtype"] = subtype
            else:
                extra_out.pop("adas_subtype", None)

    return _sanitize_uniform_scenario(out)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def _resolve_kb_path() -> Path:
    """
    After pdf_to_json_raw.build_knowledge_base() runs, resolve the KB file path.
    Prefer module-level OUTPUT_JSON if present; fallback to common names.
    """
    kb_attr = getattr(pdf_to_json_raw, "OUTPUT_JSON", None)
    if kb_attr:
        p = Path(str(kb_attr)).expanduser()
        if p.is_absolute():
            if p.exists():
                return p
            # If absolute but missing, still return it so error prints useful path
            return p
        # relative path: resolve relative to project root
        p2 = (PROJECT_ROOT / p).resolve()
        if p2.exists():
            return p2

    # Fall back to common candidates in known roots
    candidates = [
        PROJECT_ROOT / "knowledge_base_raw.json",
        PROJECT_ROOT / "knowledge_base.json",
        PROJECT_ROOT / "PARSER" / "knowledge_base_raw.json",
        PROJECT_ROOT / "PARSER" / "knowledge_base.json",
        DEFAULT_RUN_ROOT / "knowledge_base_raw.json",
        DEFAULT_RUN_ROOT / "knowledge_base.json",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # default: expected output location
    return (PROJECT_ROOT / "knowledge_base_raw.json").resolve()


def run_full_pipeline_for_pdf(
    pdf_bytes: bytes,
    filename: str,
    use_llm_parser: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full 3-stage pipeline and return uniform-schema scenarios."""

    # -------------------------
    # Stage 1: PDF -> assets
    # -------------------------
    BASE_PDF_FOLDER.mkdir(parents=True, exist_ok=True)
    pdf_path = (BASE_PDF_FOLDER / filename).resolve()
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Let parser_1 control its own output folder structure.
    if hasattr(parser_1, "prepare_output_dirs"):
        dirs = []
        for attr in (
            "TEXT_OUTPUT",
            "IMAGE_OUTPUT",
            "OCR_OUTPUT",
            "TABLE_OUTPUT",
            "PARSED_ROOT",
            "PARSED_DATA_ROOT",
            "PARSED_DATA_DIR",
        ):
            if hasattr(parser_1, attr):
                val = getattr(parser_1, attr)
                if isinstance(val, str) and val:
                    dirs.append(val)
        if dirs:
            parser_1.prepare_output_dirs(dirs)

    # Call parser_1.process_pdf (newer: process_pdf(pdf_path), older: process_pdf(pdf_path, reader))
    sig = None
    try:
        sig = inspect.signature(parser_1.process_pdf)
    except Exception:
        sig = None

    if sig is not None and len(sig.parameters) >= 2:
        if easyocr is None:
            raise ImportError(
                "easyocr is required because parser_1.process_pdf expects a reader argument. "
                "Install easyocr or update parser_1.process_pdf to accept only the PDF path."
            )
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        parser_1.process_pdf(str(pdf_path), reader)
    else:
        parser_1.process_pdf(str(pdf_path))

    # -------------------------
    # Stage 1b: build KB
    # -------------------------
    if hasattr(pdf_to_json_raw, "build_knowledge_base"):
        pdf_to_json_raw.build_knowledge_base()
    elif hasattr(pdf_to_json_raw, "main"):
        pdf_to_json_raw.main()
    else:
        raise AttributeError("pdf_to_json_raw must expose build_knowledge_base() or main()")

    kb_path = _resolve_kb_path()

    # Define ONE deterministic run folder for stage2/3 outputs.
    # Use the folder where KB lives (best), else fall back to Parsed_Data/
    run_root = kb_path.parent if kb_path.parent.exists() else DEFAULT_RUN_ROOT
    run_root.mkdir(parents=True, exist_ok=True)

    out_structured = (run_root / "structured_scenarios.json").resolve()
    out_evidence = (run_root / "scenario_evidence.json").resolve()
    out_report = (run_root / "evidence_report.json").resolve()
    out_enriched = (run_root / "enriched_scenarios.json").resolve()
    out_uniform = (run_root / "uniform_scenarios.json").resolve()

    # -------------------------
    # Stage 2: KB -> structured + evidence
    # -------------------------
    # IMPORTANT: use absolute paths so Streamlit CWD never matters.
    # Force Stage-2 to use latest env values (it reads KB_PATH at import time otherwise)
    stage2_build_structured_and_evidence.KB_PATH = str(kb_path.resolve())
    stage2_build_structured_and_evidence.OUT_STRUCTURED = str(out_structured)
    stage2_build_structured_and_evidence.OUT_EVIDENCE = str(out_evidence)
    stage2_build_structured_and_evidence.OUT_REPORT = str(out_report)



    stage2_build_structured_and_evidence.main()

    if not out_structured.exists():
        raise FileNotFoundError(
            f"Stage-2 did not produce structured output at: {out_structured}\n"
            f"KB_PATH={os.environ.get('KB_PATH')}\n"
            f"Run root={run_root}\n"
            f"Tip: check stage2_build_structured_and_evidence output paths/env handling."
        )

    # -------------------------
    # Load structured scenarios
    # -------------------------
    with open(out_structured, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    if not isinstance(scenarios, list):
        return []

    for s in scenarios:
        if isinstance(s, dict):
            _tag_adas_family(s)

    # -------------------------
    # Stage 3: LLM enrichment (optional)
    # -------------------------
    if use_llm_parser:
        # Ensure evidence path exists (even if empty)
        if not out_evidence.exists():
            # create empty evidence file rather than failing hard
            with open(out_evidence, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2, ensure_ascii=False)

        scenarios = enrich_all(
            scenarios,
            knowledge_base_path=str(kb_path.resolve()),
            scenario_evidence_path=str(out_evidence.resolve()),
        )

        with open(out_enriched, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Output: ALWAYS uniform schema + SANITIZED
    # -------------------------
    uniform_out: List[Dict[str, Any]] = []
    for s in scenarios:
        if not isinstance(s, dict):
            continue
        uniform_out.append(_build_uniform_schema(s))

    with open(out_uniform, "w", encoding="utf-8") as f:
        json.dump(uniform_out, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Stage 4: Generate Accuracy Report (NEW!)
    # -------------------------
    try:
        from PARSER.report_generator import generate_accuracy_report
        
        out_report_pdf = (run_root / "Euro_NCAP_Scenario_Analysis_Report.pdf").resolve()
        
        print(f"[main_parser] Generating accuracy report...")
        
        generate_accuracy_report(
            pdf_path=str(pdf_path.resolve()),
            scenarios_json_path=str(out_uniform.resolve()),
            output_pdf_path=str(out_report_pdf),
            protocol_version="4.3.1"
        )
        
        print(f"[main_parser] ✅ Accuracy report generated: {out_report_pdf}")
        
    except ImportError:
        print("[main_parser] ⚠️ report_generator not found - skipping report generation")
    except Exception as e:
        print(f"[main_parser] ⚠️ Report generation failed: {e}")
        # Don't fail the whole pipeline if report generation fails

    return uniform_out


if __name__ == "__main__":
    import sys

    print("[main_parser] Standalone run starting...")

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python main_parser.py <path-to-pdf>\n"
            "Example:\n"
            "  python main_parser.py ../EuroNcap/euro-ncap-aeb-c2c-test-protocol-v431.pdf"
        )

    pdf_path = Path(sys.argv[1]).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    out = run_full_pipeline_for_pdf(
        pdf_bytes=pdf_bytes,
        filename=pdf_path.name,
        use_llm_parser=False,  # set True when you want LLM enrichment
    )

    print(f"[main_parser] Done. uniform scenarios: {len(out)}")