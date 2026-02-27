"""
RAG2/generators/xosc_generator.py

FIXED VERSION:
- More flexible entity name matching (ego_vehicle, ego, ego_actor all accepted)
- Relaxed position validation (accepts various Position element types)
- Better error messages
- Improved teleport detection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import re
import xml.etree.ElementTree as ET

from ..scenario_utils import _ensure_mandatory_user_config, _family_of, _get_path
from ..chroma_store import retrieve_context
from ..llm_client import call_llm_json
from ..prompts.xosc_prompts import build_xosc_prompts

# Optional deterministic fallback (non-AEB only)
try:
    from ..xosc_builder import _build_xosc_v5
except Exception:
    _build_xosc_v5 = None  # type: ignore


_AEB_VARIANT_PATTERNS = [
    ("ccrs", re.compile(r"\bccrs\b", re.IGNORECASE)),
    ("ccrm", re.compile(r"\bccrm\b", re.IGNORECASE)),
    ("ccrb", re.compile(r"\bccrb\b", re.IGNORECASE)),
    ("cccscp", re.compile(r"\bcccscp\b", re.IGNORECASE)),
    ("ccftap", re.compile(r"\bccftap\b", re.IGNORECASE)),
    ("ccfhol", re.compile(r"\bccfhol\b", re.IGNORECASE)),
    ("ccfhos", re.compile(r"\bccfhos\b", re.IGNORECASE)),
]


def _infer_aeb_variant_key(scenario: Dict[str, Any]) -> str:
    hints = scenario.get("runtime_hints") or {}
    if isinstance(hints, dict):
        v = hints.get("aeb_variant")
        if isinstance(v, str) and v.strip():
            return v.strip().lower()

    v2 = _get_path(scenario, "user_config.behavior.scenario_variant", None)
    if isinstance(v2, str) and v2.strip():
        return v2.strip().lower()

    v3 = _get_path(scenario, "classification.variant", None)
    if isinstance(v3, str) and v3.strip():
        return v3.strip().lower()

    scenario_name = (scenario.get("scenario_name") or scenario.get("name") or "").strip()
    for key, pat in _AEB_VARIANT_PATTERNS:
        if pat.search(scenario_name):
            return key
    return "unknown"


def _normalize_trigger_type(scenario: Dict[str, Any]) -> str:
    t = _get_path(scenario, "user_config.trigger.type", "")
    if not isinstance(t, str):
        t = str(t)
    return t.strip().upper()


def _augment_user_prompt_with_errors(user_prompt: str, errors: List[str]) -> str:
    err_blob = "\n".join([f"- {e}" for e in errors])
    return (
        user_prompt
        + "\n\n"
        + "=== VALIDATION ERRORS FROM LAST OUTPUT (MUST FIX ALL) ===\n"
        + err_blob
        + "\n\n"
        + "Regenerate the FULL OpenSCENARIO XML to satisfy all requirements.\n"
        + "Return ONLY JSON: {\"xosc\": \"...\"}\n"
        + "The XML MUST include a comment tag like:\n"
        + "<!-- GENERATED_BY: <model_name> -->\n"
        + "\n"
        + "CRITICAL COMPLETENESS:\n"
        + "- In <Storyboard><Init><Actions>, include TeleportAction + Position for ego and target.\n"
        + "- Position must be LanePosition or WorldPosition.\n"
        + "- Include Init AbsoluteTargetSpeed for ego+target.\n"
        + "- Trigger must match user_config.trigger.type:\n"
        + "  START_IMMEDIATELY => SimulationTimeCondition\n"
        + "  DISTANCE => RelativeDistanceCondition\n"
        + "  TTC => TimeToCollisionCondition\n"
        + "- Always include StopTrigger with timeout.\n"
        + "\n"
        + "STRUCTURE RULE:\n"
        + "- Under <Actions>, use <Private entityRef=\"...\"> ... <PrivateAction> ... </Private>.\n"
        + "- Do NOT output floating <PrivateAction> directly under <Actions>.\n"
        + "- Entity names can be 'ego', 'ego_vehicle', 'target', or 'target_vehicle'.\n"
    )


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(v: Any, default: float) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _et_parse_keep_comments(xml_text: str) -> ET.Element:
    """
    Parse XML while preserving comments (Python 3.8+ supports insert_comments=True).
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.fromstring(xml_text, parser=parser)


def _find_first(root: ET.Element, path: str) -> Optional[ET.Element]:
    return root.find(path)


def _tag_endswith(elem: ET.Element, suffix: str) -> bool:
    return elem.tag.endswith(suffix)


def _ensure_private_with_teleport(
    actions_elem: ET.Element,
    entity_ref: str,
    *,
    world_x: float,
    world_y: float,
    world_z: float,
) -> None:
    """
    Ensure there is a <Private entityRef="X"> block under Actions that contains
    a TeleportAction->Position->WorldPosition (or LanePosition).
    If Private exists but teleport is missing, add teleport.
    If Private doesn't exist, create it with teleport.
    If teleport exists already, do nothing (avoid double-teleport).
    """
    private_elem: Optional[ET.Element] = None
    for child in list(actions_elem):
        if _tag_endswith(child, "Private") and child.attrib.get("entityRef") == entity_ref:
            private_elem = child
            break

    if private_elem is None:
        private_elem = ET.Element("Private", {"entityRef": entity_ref})
        actions_elem.insert(0, private_elem)  # prepend so placement happens early

    # If there is already any TeleportAction under this Private, do not add another
    if private_elem.find(".//TeleportAction") is not None:
        return

    pa = ET.SubElement(private_elem, "PrivateAction")
    ta = ET.SubElement(pa, "TeleportAction")
    pos = ET.SubElement(ta, "Position")
    ET.SubElement(
        pos,
        "WorldPosition",
        {
            "x": str(world_x),
            "y": str(world_y),
            "z": str(world_z),
            "h": "0",
            "p": "0",
            "r": "0",
        },
    )


def _sanitize_init_actions_and_patch_teleports(xosc_xml: str, scenario: Dict[str, Any]) -> str:
    """
    Deterministic post-processor:
    - Removes invalid floating <PrivateAction> directly under <Actions>
    - Ensures ego & target have TeleportAction+Position inside <Private entityRef="...">
    - Avoids double-teleport by not adding if any TeleportAction already exists per actor
    """
    xml = xosc_xml.strip()
    if not xml:
        return xml

    try:
        root = _et_parse_keep_comments(xml)
    except ET.ParseError:
        return xml  # validator will surface parse error

    actions = _find_first(root, ".//Storyboard/Init/Actions")
    if actions is None:
        return xml  # validator will handle

    # 1) Remove invalid direct children like <PrivateAction> under <Actions>
    for child in list(actions):
        if _tag_endswith(child, "PrivateAction"):
            actions.remove(child)

    # 2) Ensure teleport for ego and target (with fallback positions)
    gap_m = _safe_float(_get_path(scenario, "user_config.layout.initial_gap_m", 15.0), 15.0)
    lat_m = _safe_float(_get_path(scenario, "user_config.layout.lateral_offset_m", 0.0), 0.0)

    # Try multiple entity name variants
    ego_names = ["ego", "ego_vehicle", "ego_actor"]
    target_names = ["target", "target_vehicle", "target_actor"]
    
    # Find which ego name is actually used
    ego_ref = "ego"
    for name in ego_names:
        if f'entityRef="{name}"' in xml or f"ScenarioObject name=\"{name}\"" in xml:
            ego_ref = name
            break
    
    # Find which target name is actually used
    target_ref = "target"
    for name in target_names:
        if f'entityRef="{name}"' in xml or f"ScenarioObject name=\"{name}\"" in xml:
            target_ref = name
            break

    _ensure_private_with_teleport(actions, ego_ref, world_x=0.0, world_y=lat_m, world_z=0.5)
    _ensure_private_with_teleport(actions, target_ref, world_x=gap_m, world_y=lat_m, world_z=0.5)

    # Optional VRU teleport if vru blueprint exists
    vru_bp = _get_path(scenario, "user_config.entities.vru.blueprint", None)
    if vru_bp:
        _ensure_private_with_teleport(actions, "vru", world_x=gap_m * 0.5, world_y=lat_m + 2.0, world_z=0.5)

    # Serialize back
    out = ET.tostring(root, encoding="unicode")
    # Preserve XML declaration if present originally
    if xml.startswith("<?xml"):
        if not out.startswith("<?xml"):
            out = '<?xml version="1.0" encoding="UTF-8"?>\n' + out
    return out


def _has_init_teleport_for(xml: str, entity_names: List[str]) -> bool:
    """
    FIXED: Checks whether TeleportAction+Position exists for any of the given entity names.
    More flexible - accepts ego, ego_vehicle, ego_actor, etc.
    """
    for entity_name in entity_names:
        # Look for Private blocks with this entity ref
        patterns = [
            rf'<Private\s+entityRef="{re.escape(entity_name)}".*?</Private>',
            rf'<Private\s+entityRef=\'{re.escape(entity_name)}\'.*?</Private>',
        ]
        
        for pattern in patterns:
            pat = re.compile(pattern, re.DOTALL)
            m = pat.search(xml)
            if m:
                blk = m.group(0)
                # Check if this block has TeleportAction and Position
                has_teleport = "TeleportAction" in blk
                has_position = "<Position" in blk
                has_position_type = any(pos_type in blk for pos_type in [
                    "WorldPosition", "LanePosition", "RelativeWorldPosition", 
                    "RelativeLanePosition", "RoadPosition"
                ])
                
                if has_teleport and has_position and has_position_type:
                    return True
    
    return False


def validate_aeb_xosc(xosc_xml: str, scenario: Dict[str, Any]) -> List[str]:
    """
    FIXED: AEB completeness validator with relaxed entity name matching.
    """
    errors: List[str] = []

    if not isinstance(xosc_xml, str) or not xosc_xml.strip():
        return ["Empty or non-string 'xosc' output."]

    xml = xosc_xml.strip()

    if "GENERATED_BY:" not in xml:
        errors.append("Missing required XML comment marker 'GENERATED_BY:'.")

    try:
        root = _et_parse_keep_comments(xml)
    except ET.ParseError as e:
        errors.append(f"XML parse error: {str(e)}")
        return errors

    if "OpenSCENARIO" not in root.tag:
        errors.append("Root element must be <OpenSCENARIO>.")
        return errors

    if root.find(".//Storyboard") is None:
        errors.append("Missing <Storyboard> element.")

    entities = root.find(".//Entities")
    if entities is None:
        errors.append("Missing <Entities> element.")
        return errors

    # FIXED: Check for ego with flexible naming
    ego_patterns = ['name="ego"', 'name="ego_vehicle"', 'name="ego_actor"']
    if not any(pattern in xml for pattern in ego_patterns):
        errors.append('Missing ScenarioObject with name="ego" (or ego_vehicle/ego_actor).')
    
    # FIXED: Check for target with flexible naming
    target_patterns = ['name="target"', 'name="target_vehicle"', 'name="target_actor"']
    if not any(pattern in xml for pattern in target_patterns):
        errors.append('Missing ScenarioObject with name="target" (or target_vehicle/target_actor).')

    if "AbsoluteTargetSpeed" not in xml and "SpeedAction" not in xml:
        errors.append("Missing speed actions (AbsoluteTargetSpeed or SpeedAction) for ego/target in Init.")

    # FIXED: Use flexible entity name lists
    ego_names = ["ego", "ego_vehicle", "ego_actor"]
    target_names = ["target", "target_vehicle", "target_actor"]
    
    if not _has_init_teleport_for(xml, ego_names):
        errors.append("Missing TeleportAction+Position for ego in Init.")
    if not _has_init_teleport_for(xml, target_names):
        errors.append("Missing TeleportAction+Position for target in Init.")

    trig_type = _normalize_trigger_type(scenario)
    has_sim_time = "SimulationTimeCondition" in xml
    has_dist = "RelativeDistanceCondition" in xml or "DistanceCondition" in xml
    has_ttc = "TimeToCollisionCondition" in xml or "TimeHeadway" in xml

    if trig_type == "START_IMMEDIATELY":
        if not has_sim_time:
            errors.append("Trigger type START_IMMEDIATELY but missing SimulationTimeCondition.")
    elif trig_type == "DISTANCE":
        if not has_dist:
            errors.append("Trigger type DISTANCE but missing RelativeDistanceCondition.")
    elif trig_type == "TTC":
        if not has_ttc:
            errors.append("Trigger type TTC but missing TimeToCollisionCondition.")
    else:
        # unknown trigger type: don't block
        pass

    if "StopTrigger" not in xml:
        errors.append("Missing StopTrigger. Must include timeout-based stop condition.")

    # Structure sanity: no floating PrivateAction directly under Init/Actions
    try:
        actions = root.find(".//Storyboard/Init/Actions")
        if actions is not None:
            for child in list(actions):
                if _tag_endswith(child, "PrivateAction"):
                    errors.append("Invalid structure: floating <PrivateAction> directly under <Actions>. Must be wrapped in <Private entityRef=\"...\">.")
                    break
    except Exception:
        pass

    return errors


def generate_xosc_rag(
    scenario: Dict[str, Any],
    *,
    k: int | None = None,
    provider: str = "openai",
    enable_fallback_builder: bool = True,
) -> str:
    """
    FIXED: Generate XOSC via RAG + LLM with improved validation.
    """
    if not isinstance(scenario, dict):
        raise RuntimeError("generate_xosc_rag expects a scenario dict.")

    family = _family_of(scenario)
    _ensure_mandatory_user_config(scenario)

    scenario_name = (scenario.get("scenario_name") or scenario.get("name") or "").strip()
    trig_type = _normalize_trigger_type(scenario)
    town = _get_path(scenario, "user_config.map.town", None)
    road_mode = _get_path(scenario, "user_config.map.road_selection_mode", None)

    if family.upper() == "AEB":
        enable_fallback_builder = False
        aeb_variant = _infer_aeb_variant_key(scenario)
        query_text = (
            "OpenSCENARIO 1.0 XOSC complete examples with TeleportAction and LanePosition/WorldPosition, "
            "complete Init + StartTrigger + StopTrigger, and CARLA blueprint properties. "
            "AEB scenarios (rear stationary/moving/braking/cut-in) and trigger patterns "
            "(START_IMMEDIATELY, DISTANCE, TTC). "
            f"AEB variant={aeb_variant}. TriggerType={trig_type or 'UNKNOWN'}. "
            f"Town={town or 'UNKNOWN'} RoadMode={road_mode or 'UNKNOWN'}. "
            f"Scenario={scenario_name}"
        )
    else:
        query_text = (
            "OpenSCENARIO 1.0 XOSC templates with TeleportAction and LanePosition/WorldPosition, "
            "complete Init, triggers, stop conditions. "
            f"family={family}. TriggerType={trig_type or 'UNKNOWN'}. Scenario={scenario_name}"
        )

    retrieved: List[Dict[str, Any]] = retrieve_context(query_text, k=k)
    system_prompt, user_prompt = build_xosc_prompts(
        scenario=scenario,
        family=family,
        retrieved_context=retrieved,
    )

    # AEB: validate + retry once (patch before validate)
    if family.upper() == "AEB":
        max_tries = 2
        last_errors: List[str] = []
        prompt_for_attempt = user_prompt

        for attempt in range(max_tries):
            result = call_llm_json(system_prompt, prompt_for_attempt, provider=provider)

            xosc = result.get("xosc")
            if not isinstance(xosc, str) or not xosc.strip():
                last_errors = ["LLM output JSON did not contain a non-empty 'xosc' string."]
                prompt_for_attempt = _augment_user_prompt_with_errors(user_prompt, last_errors)
                continue

            # PATCH: sanitize Actions structure + ensure teleports
            xosc2 = _sanitize_init_actions_and_patch_teleports(xosc, scenario)

            errors = validate_aeb_xosc(xosc2, scenario)
            
            # On last attempt, be more lenient
            if attempt == max_tries - 1 and errors:
                # Filter out soft warnings
                critical_errors = [e for e in errors if not any(soft in e.lower() for soft in [
                    "missing required xml comment",  # Comment is nice-to-have
                ])]
                
                if not critical_errors:
                    return xosc2.strip()
                
                last_errors = critical_errors
            elif not errors:
                return xosc2.strip()
            else:
                last_errors = errors
            
            prompt_for_attempt = _augment_user_prompt_with_errors(user_prompt, errors)

        raise RuntimeError(
            "AEB XOSC generation failed after retries.\n"
            + "\n".join([f"- {e}" for e in last_errors])
        )

    # Non-AEB: one-shot + optional fallback
    result = call_llm_json(system_prompt, user_prompt, provider=provider)

    xosc = result.get("xosc")
    if isinstance(xosc, str) and xosc.strip():
        return xosc.strip()

    if enable_fallback_builder and _build_xosc_v5 is not None:
        return _build_xosc_v5(scenario)

    raise RuntimeError("LLM output JSON did not contain a non-empty 'xosc' string and fallback is disabled.")