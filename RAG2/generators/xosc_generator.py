"""
RAG2/generators/xosc_generator.py
FIXED VERSION v4:
- Fixed CCRb/CCRm spawn positions (hero behind s=156.84, adversary ahead s=193.66)
- Added all VRU scenarios to _LANE_POSITIONS (pedestrian, cyclist, motorcyclist)
- Added VRU variant patterns to _AEB_VARIANT_PATTERNS
- All other fixes from v3 preserved
- Validated: CARLA 0.9.15 / ScenarioRunner 0.9.16 / Town01
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import re
import xml.etree.ElementTree as ET
from ..scenario_utils import _ensure_mandatory_user_config, _family_of, _get_path
from ..chroma_store import retrieve_context
from ..llm_client import call_llm_json
from ..prompts.xosc_prompts import build_xosc_prompts

try:
    from ..xosc_builder import _build_xosc_v5
except Exception:
    _build_xosc_v5 = None  # type: ignore

_AEB_VARIANT_PATTERNS = [
    # AEB C2C scenarios
    ("ccrs",    re.compile(r"\bccrs\b",    re.IGNORECASE)),
    ("ccrm",    re.compile(r"\bccrm\b",    re.IGNORECASE)),
    ("ccrb",    re.compile(r"\bccrb\b",    re.IGNORECASE)),
    ("cccscp",  re.compile(r"\bcccscp\b",  re.IGNORECASE)),
    ("ccftap",  re.compile(r"\bccftap\b",  re.IGNORECASE)),
    ("ccftab",  re.compile(r"\bccftab\b",  re.IGNORECASE)),
    ("ccfhol",  re.compile(r"\bccfhol\b",  re.IGNORECASE)),
    ("ccfhos",  re.compile(r"\bccfhos\b",  re.IGNORECASE)),
    ("ccfho",   re.compile(r"\bccfho\b",   re.IGNORECASE)),
    # VRU pedestrian scenarios
    ("cpfa",    re.compile(r"\bcpfa\b",    re.IGNORECASE)),
    ("cpna",    re.compile(r"\bcpna\b",    re.IGNORECASE)),
    ("cpla",    re.compile(r"\bcpla\b",    re.IGNORECASE)),
    ("cpnco",   re.compile(r"\bcpnco\b",   re.IGNORECASE)),
    ("cpta",    re.compile(r"\bcpta\b",    re.IGNORECASE)),
    ("cpra",    re.compile(r"\bcpra\b",    re.IGNORECASE)),
    ("cpla",    re.compile(r"\bcpla\b",    re.IGNORECASE)),
    # VRU cyclist scenarios
    ("cbna",    re.compile(r"\bcbna\b",    re.IGNORECASE)),
    ("cbfa",    re.compile(r"\bcbfa\b",    re.IGNORECASE)),
    ("cbla",    re.compile(r"\bcbla\b",    re.IGNORECASE)),
    ("cbta",    re.compile(r"\bcbta\b",    re.IGNORECASE)),
    ("cbnao",   re.compile(r"\bcbnao\b",   re.IGNORECASE)),
    ("cbda",    re.compile(r"\bcbda\b",    re.IGNORECASE)),
    # VRU motorcyclist scenarios
    ("cmrs",    re.compile(r"\bcmrs\b",    re.IGNORECASE)),
    ("cmrb",    re.compile(r"\bcmrb\b",    re.IGNORECASE)),
    ("cmftap",  re.compile(r"\bcmftap\b",  re.IGNORECASE)),
    ("cmoncoming",   re.compile(r"\bcmoncoming\b",   re.IGNORECASE)),
    ("cmovertaking", re.compile(r"\bcmovertaking\b", re.IGNORECASE)),
]

# Verified Town01 spawn positions per scenario
# ALL values validated on CARLA 0.9.15 / SR 0.9.16 / Town01
_LANE_POSITIONS = {

    # ── AEB C2C REAR SCENARIOS ────────────────────────────────────────────────
    # Hero BEHIND (lower s), adversary AHEAD (higher s)
    "ccrs": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "ccrm": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "ccrb": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },

    # ── AEB C2C FRONT CROSSING SCENARIOS ─────────────────────────────────────
    "ccftap": {
        "hero":     {"roadId": "4",  "laneId": "-1", "offset": "0.0", "s": "197.98"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "ccftab": {
        "hero":     {"roadId": "4",  "laneId": "-1", "offset": "0.0", "s": "197.98"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },

    # ── AEB C2C HEAD-ON SCENARIOS ─────────────────────────────────────────────
    # Adversary on opposite lane (laneId=1)
    "ccfhos": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId":  "1", "offset": "0.0", "s": "193.66"},
    },
    "ccfhol": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId":  "1", "offset": "0.0", "s": "193.66"},
    },
    "ccfho": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId":  "1", "offset": "0.0", "s": "193.66"},
    },

    # ── VRU PEDESTRIAN SCENARIOS ──────────────────────────────────────────────
    # Pedestrian ahead of hero, same lane
    "cpfa": {
        "hero":        {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":   {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
        "pedestrian":  {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cpna": {
        "hero":        {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":   {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
        "pedestrian":  {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cpla": {
        "hero":        {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":   {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
        "pedestrian":  {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cpnco": {
        "hero":        {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":   {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
        "pedestrian":  {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cpta": {
        "hero":        {"roadId": "4",  "laneId": "-1", "offset": "0.0", "s": "197.98"},
        "adversary":   {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
        "pedestrian":  {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cpra": {
        "hero":        {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
        "adversary":   {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "pedestrian":  {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
    },
    "cpla": {
        "hero":        {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":   {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
        "pedestrian":  {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },

    # ── VRU CYCLIST SCENARIOS ─────────────────────────────────────────────────
    "cbna": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cbfa": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cbla": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cbta": {
        "hero":     {"roadId": "4",  "laneId": "-1", "offset": "0.0", "s": "197.98"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cbnao": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cbda": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },

    # ── VRU MOTORCYCLIST SCENARIOS ────────────────────────────────────────────
    "cmrs": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cmrb": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cmftap": {
        "hero":     {"roadId": "4",  "laneId": "-1", "offset": "0.0", "s": "197.98"},
        "adversary":{"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "193.66"},
    },
    "cmoncoming": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId":  "1", "offset": "0.0", "s": "193.66"},
    },
    "cmovertaking": {
        "hero":     {"roadId": "12", "laneId": "-1", "offset": "0.0", "s": "156.84"},
        "adversary":{"roadId": "12", "laneId":  "1", "offset": "0.0", "s": "193.66"},
    },
}

_FRONT_SCENARIOS = {
    "ccftap", "ccftab", "ccfhos", "ccfhol", "ccfho",
    "cmftap", "cmoncoming", "cmovertaking",
    "cpfa", "cpna", "cplo", "cpnco", "cpta", "cpla",
    "cbna", "cbfa", "cbla", "cbta", "cbnao", "cbda",
}


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
    # Also check scenario_code
    v4 = _get_path(scenario, "scenario_code", None)
    if isinstance(v4, str) and v4.strip():
        code = v4.strip().lower()
        # Strip overlap suffix e.g. cpna-25 -> cpna
        code = re.sub(r"[-/]\d+$", "", code)
        for key, pat in _AEB_VARIANT_PATTERNS:
            if pat.search(code):
                return key
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


def _fix_init_speed(xosc_xml: str) -> str:
    """
    Force replace $heroSpeed/$adversarySpeed ONLY inside <Init>...</Init> block.
    Story ManeuverGroup speeds are left untouched.
    """
    def fix_init_block(match):
        block = match.group(0)
        block = re.sub(
            r'(<AbsoluteTargetSpeed\s+value=")[^"]*(\$heroSpeed|hero_speed|heroSpeed)[^"]*(")',
            r'\g<1>0.0\3',
            block
        )
        block = re.sub(
            r'(<AbsoluteTargetSpeed\s+value=")[^"]*(\$adversarySpeed|adversary_speed|adversarySpeed)[^"]*(")',
            r'\g<1>0.0\3',
            block
        )
        return block
    return re.sub(r'<Init>.*?</Init>', fix_init_block, xosc_xml, flags=re.DOTALL)


def _fix_hero_type(xosc_xml: str) -> str:
    """
    Fix hero entity Property type from 'simulation' to 'ego_vehicle'.
    Only applies to the hero ScenarioObject block — does not touch adversary/pedestrian.
    """
    def fix_hero_block(match):
        block = match.group(0)
        block = block.replace(
            '<Property name="type" value="simulation"',
            '<Property name="type" value="ego_vehicle"'
        )
        return block

    return re.sub(
        r'<ScenarioObject\s+name="hero">.*?</ScenarioObject>',
        fix_hero_block,
        xosc_xml,
        flags=re.DOTALL
    )


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
        + "- In <Storyboard><Init><Actions>, include TeleportAction + Position for hero and adversary.\n"
        + "- Position MUST use LanePosition with verified Town01 spawn points.\n"
        + "- NEVER use WorldPosition or RelativeRoadPosition.\n"
        + "- CCR scenarios: hero roadId='12' laneId='-1' s='156.84', adversary roadId='12' laneId='-1' s='193.66'.\n"
        + "- CCFhos/CCFhol: hero laneId='-1' s='156.84', adversary laneId='1' s='193.66'.\n"
        + "- CCFtap/CCFtab: hero roadId='4' laneId='-1' s='197.98', adversary roadId='12' laneId='-1' s='193.66'.\n"
        + "- Init AbsoluteTargetSpeed MUST be 0.0 for ALL entities — NEVER $heroSpeed or $adversarySpeed.\n"
        + "- Story ManeuverGroup SpeedAction MUST use value='$heroSpeed' / value='$adversarySpeed'.\n"
        + "- Story SpeedAction MUST use dynamicsShape='linear' value='3.0' dynamicsDimension='time'.\n"
        + "- SimulationTimeCondition value MUST always be 1.0 (never 0.1 or 0.0).\n"
        + "- AEB trigger CCR: cartesianDistance value=12.0 freespace='false'.\n"
        + "- AEB trigger CCF/VRU: cartesianDistance value=20.0 freespace='false'.\n"
        + "- AEB brake: dynamicsShape='linear' value='3.0' dynamicsDimension='time'.\n"
        + "- Always include WaitGroup ManeuverGroup triggering at t=55s.\n"
        + "- Global StopTrigger: criteria_CollisionTest parameterRef='criteria_CollisionTest'.\n"
        + "- Do NOT include criteria_DrivenDistanceTest.\n"
        + "- Trigger START_IMMEDIATELY => SimulationTimeCondition value=1.0\n"
        + "- Always include StopTrigger with 60s timeout.\n"
        + "\n"
        + "STRUCTURE RULE:\n"
        + "- Under <Actions>, use <Private entityRef='...'> ... <PrivateAction> ... </Private>.\n"
        + "- Do NOT output floating <PrivateAction> directly under <Actions>.\n"
        + "- Entity names MUST be 'hero' and 'adversary'.\n"
    )


def _safe_float(v: Any, default: float) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _et_parse_keep_comments(xml_text: str) -> ET.Element:
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
    aeb_variant: str = "ccrs",
    entity_type: str = "hero",
) -> None:
    """
    Ensure TeleportAction+LanePosition exists for entity in Init/Actions.
    Uses verified Town01 spawn positions from _LANE_POSITIONS.
    """
    private_elem: Optional[ET.Element] = None
    for child in list(actions_elem):
        if _tag_endswith(child, "Private") and child.attrib.get("entityRef") == entity_ref:
            private_elem = child
            break

    if private_elem is None:
        private_elem = ET.Element("Private", {"entityRef": entity_ref})
        actions_elem.insert(0, private_elem)

    if private_elem.find(".//TeleportAction") is not None:
        return

    # Get verified lane position
    variant_positions = _LANE_POSITIONS.get(aeb_variant, _LANE_POSITIONS["ccrs"])
    # For pedestrian entity type, fall back to adversary position
    lane_pos = variant_positions.get(entity_type) or variant_positions.get("adversary") or variant_positions["hero"]

    pa = ET.SubElement(private_elem, "PrivateAction")
    ta = ET.SubElement(pa, "TeleportAction")
    pos = ET.SubElement(ta, "Position")
    ET.SubElement(
        pos,
        "LanePosition",
        {
            "roadId": lane_pos["roadId"],
            "laneId": lane_pos["laneId"],
            "offset": lane_pos["offset"],
            "s":      lane_pos["s"],
        },
    )


def _sanitize_init_actions_and_patch_teleports(xosc_xml: str, scenario: Dict[str, Any]) -> str:
    """
    Deterministic post-processor:
    - Removes floating <PrivateAction> directly under <Actions>
    - Ensures hero & adversary have TeleportAction+LanePosition in Init
    - Forces init speed to 0.0 (Init block only)
    """
    xml = xosc_xml.strip()
    if not xml:
        return xml

    xml = _fix_init_speed(xml)

    try:
        root = _et_parse_keep_comments(xml)
    except ET.ParseError:
        return xml

    actions = _find_first(root, ".//Storyboard/Init/Actions")
    if actions is None:
        return xml

    for child in list(actions):
        if _tag_endswith(child, "PrivateAction"):
            actions.remove(child)

    aeb_variant = _infer_aeb_variant_key(scenario)

    ego_names = ["ego", "ego_vehicle", "ego_actor", "hero"]
    target_names = ["target", "target_vehicle", "target_actor", "adversary"]

    ego_ref = "hero"
    for name in ego_names:
        if f'entityRef="{name}"' in xml or f'ScenarioObject name="{name}"' in xml:
            ego_ref = name
            break

    target_ref = "adversary"
    for name in target_names:
        if f'entityRef="{name}"' in xml or f'ScenarioObject name="{name}"' in xml:
            target_ref = name
            break

    _ensure_private_with_teleport(actions, ego_ref,    aeb_variant=aeb_variant, entity_type="hero")

    # Only inject adversary teleport if "adversary" entity exists in <Entities>
    # For pedestrian scenarios, entity is named "pedestrian" not "adversary"
    adversary_exists = 'ScenarioObject name="adversary"' in xml or 'ScenarioObject name="target"' in xml
    if adversary_exists:
        _ensure_private_with_teleport(actions, target_ref, aeb_variant=aeb_variant, entity_type="adversary")

    # Handle pedestrian entity if present (instead of adversary)
    pedestrian_exists = 'ScenarioObject name="pedestrian"' in xml
    if pedestrian_exists:
        _ensure_private_with_teleport(actions, "pedestrian", aeb_variant=aeb_variant, entity_type="pedestrian")

    out = ET.tostring(root, encoding="unicode")
    if xml.startswith("<?xml") and not out.startswith("<?xml"):
        out = '<?xml version="1.0" encoding="UTF-8"?>\n' + out

    out = _fix_init_speed(out)
    out = _fix_hero_type(out)
    return out


def _has_init_teleport_for(xml: str, entity_names: List[str]) -> bool:
    for entity_name in entity_names:
        patterns = [
            rf'<Private\s+entityRef="{re.escape(entity_name)}".*?</Private>',
            rf'<Private\s+entityRef=\'{re.escape(entity_name)}\'.*?</Private>',
        ]
        for pattern in patterns:
            pat = re.compile(pattern, re.DOTALL)
            m = pat.search(xml)
            if m:
                blk = m.group(0)
                has_teleport = "TeleportAction" in blk
                has_position = "<Position" in blk
                has_position_type = any(pos_type in blk for pos_type in [
                    "WorldPosition", "LanePosition", "RelativeWorldPosition",
                    "RelativeLanePosition", "RoadPosition", "RelativeRoadPosition"
                ])
                if has_teleport and has_position and has_position_type:
                    return True
    return False


def validate_aeb_xosc(xosc_xml: str, scenario: Dict[str, Any]) -> List[str]:
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

    ego_patterns = ['name="ego"', 'name="ego_vehicle"', 'name="ego_actor"', 'name="hero"']
    if not any(pattern in xml for pattern in ego_patterns):
        errors.append('Missing ScenarioObject with name="ego" (or ego_vehicle/ego_actor/hero).')

    target_patterns = [
        'name="target"', 'name="target_vehicle"', 'name="target_actor"',
        'name="adversary"', 'name="pedestrian"'
    ]
    if not any(pattern in xml for pattern in target_patterns):
        errors.append('Missing ScenarioObject with name="target" (or adversary/pedestrian).')

    if "AbsoluteTargetSpeed" not in xml and "SpeedAction" not in xml:
        errors.append("Missing speed actions for entities in Init.")

    # Validate LanePosition used — not WorldPosition or RelativeRoadPosition
    if "WorldPosition" in xml:
        errors.append("WorldPosition detected — MUST use LanePosition for all entity spawning.")
    if "RelativeRoadPosition" in xml:
        errors.append("RelativeRoadPosition detected — MUST use LanePosition for all entity spawning.")

    # Validate SimulationTimeCondition = 1.0
    if 'SimulationTimeCondition value="0.1"' in xml or "SimulationTimeCondition value='0.1'" in xml:
        errors.append("SimulationTimeCondition value must be 1.0 not 0.1.")

    # Validate init speed = 0.0
    init_match = re.search(r'<Init>.*?</Init>', xml, re.DOTALL)
    if init_match:
        init_block = init_match.group(0)
        if 'AbsoluteTargetSpeed value="$heroSpeed"' in init_block:
            errors.append("Init AbsoluteTargetSpeed must be 0.0 not $heroSpeed.")
        if 'AbsoluteTargetSpeed value="$adversarySpeed"' in init_block:
            errors.append("Init AbsoluteTargetSpeed must be 0.0 not $adversarySpeed.")

    ego_names    = ["ego", "ego_vehicle", "ego_actor", "hero"]
    target_names = ["target", "target_vehicle", "target_actor", "adversary", "pedestrian"]

    if not _has_init_teleport_for(xml, ego_names):
        errors.append("Missing TeleportAction+Position for ego/hero in Init.")
    if not _has_init_teleport_for(xml, target_names):
        errors.append("Missing TeleportAction+Position for target/adversary/pedestrian in Init.")

    trig_type = _normalize_trigger_type(scenario)
    has_sim_time = "SimulationTimeCondition" in xml
    has_dist     = "RelativeDistanceCondition" in xml or "DistanceCondition" in xml
    has_ttc      = "TimeToCollisionCondition" in xml or "TimeHeadway" in xml

    if trig_type == "START_IMMEDIATELY":
        if not has_sim_time:
            errors.append("Trigger type START_IMMEDIATELY but missing SimulationTimeCondition.")
    elif trig_type == "DISTANCE":
        if not has_dist:
            errors.append("Trigger type DISTANCE but missing RelativeDistanceCondition.")
    elif trig_type == "TTC":
        if not has_ttc:
            errors.append("Trigger type TTC but missing TimeToCollisionCondition.")

    if "StopTrigger" not in xml:
        errors.append("Missing StopTrigger.")

    # CCFhol specific: must use FollowTrajectoryAction not LaneChangeAction
    aeb_variant = _infer_aeb_variant_key(scenario)
    if aeb_variant == "ccfhol":
        if "LaneChangeAction" in xml:
            errors.append("CCFhol: LaneChangeAction detected — causes segfault in CARLA 0.9.15. Use FollowTrajectoryAction instead.")
        if "FollowTrajectoryAction" not in xml:
            errors.append("CCFhol: Missing FollowTrajectoryAction for adversary lane change.")
        if "TimeReference" not in xml:
            errors.append("CCFhol: Missing TimeReference inside FollowTrajectoryAction — SR throws IndexError without it.")
        if xml.count("<Vertex") < 4:
            errors.append("CCFhol: FollowTrajectoryAction Polyline needs at least 4 Vertex elements.")

    try:
        actions = root.find(".//Storyboard/Init/Actions")
        if actions is not None:
            for child in list(actions):
                if _tag_endswith(child, "PrivateAction"):
                    errors.append(
                        "Invalid structure: floating <PrivateAction> directly under <Actions>. "
                        "Must be wrapped in <Private entityRef='...'>."
                    )
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
    if not isinstance(scenario, dict):
        raise RuntimeError("generate_xosc_rag expects a scenario dict.")

    family = _family_of(scenario)
    _ensure_mandatory_user_config(scenario)

    scenario_name = (scenario.get("scenario_name") or scenario.get("name") or "").strip()
    trig_type     = _normalize_trigger_type(scenario)
    town          = _get_path(scenario, "user_config.map.town", None)
    road_mode     = _get_path(scenario, "user_config.map.road_selection_mode", None)

    if family.upper() in ("AEB", "VRU"):
        enable_fallback_builder = False
        aeb_variant = _infer_aeb_variant_key(scenario)
        query_text = (
            "OpenSCENARIO 1.0 XOSC complete examples with TeleportAction and LanePosition, "
            "complete Init + StartTrigger + StopTrigger, CARLA blueprint properties, "
            "AEB/VRU scenarios with WaitGroup and criteria_CollisionTest. "
            f"variant={aeb_variant}. TriggerType={trig_type or 'UNKNOWN'}. "
            f"Town={town or 'UNKNOWN'} RoadMode={road_mode or 'UNKNOWN'}. "
            f"Scenario={scenario_name}"
        )
    else:
        query_text = (
            "OpenSCENARIO 1.0 XOSC templates with TeleportAction and LanePosition, "
            "complete Init, triggers, stop conditions. "
            f"family={family}. TriggerType={trig_type or 'UNKNOWN'}. Scenario={scenario_name}"
        )

    retrieved: List[Dict[str, Any]] = retrieve_context(query_text, k=k)
    system_prompt, user_prompt = build_xosc_prompts(
        scenario=scenario,
        family=family,
        retrieved_context=retrieved,
    )

    # CCFhol SPECIAL CASE: inject dedicated FollowTrajectoryAction prompt
    if family.upper() in ("AEB", "VRU") and _infer_aeb_variant_key(scenario) == "ccfhol":
        system_prompt = system_prompt + """

=== CCFhol SPECIAL RULES — READ CAREFULLY ===

This is a Car-to-Car Front Head-On Lane Change (CCFhol) scenario.
It has ONE critical difference from all other scenarios:

ADVERSARY MUST USE FollowTrajectoryAction — NOT LaneChangeAction.
LaneChangeAction causes a SEGMENTATION FAULT in CARLA 0.9.15. NEVER use it.

The adversary starts in the OPPOSITE lane (laneId=1) and changes into hero lane (laneId=-1)
using FollowTrajectoryAction with a Polyline of 4 vertices.

MANDATORY FollowTrajectoryAction structure for adversary:

<FollowTrajectoryAction>
  <Trajectory name="LaneChangePath" closed="false">
    <ParameterDeclarations/>
    <Shape>
      <Polyline>
        <Vertex time="0.0">
          <Position><LanePosition roadId="12" laneId="1" s="193.66" offset="0.0"/></Position>
        </Vertex>
        <Vertex time="1.5">
          <Position><LanePosition roadId="12" laneId="1" s="188.0" offset="-1.75"/></Position>
        </Vertex>
        <Vertex time="3.0">
          <Position><LanePosition roadId="12" laneId="-1" s="182.0" offset="0.0"/></Position>
        </Vertex>
        <Vertex time="10.0">
          <Position><LanePosition roadId="12" laneId="-1" s="170.0" offset="0.0"/></Position>
        </Vertex>
      </Polyline>
    </Shape>
  </Trajectory>
  <TimeReference>
    <Timing domainAbsoluteRelative="absolute" scale="1.0" offset="0.0"/>
  </TimeReference>
  <TrajectoryFollowingMode followingMode="position"/>
</FollowTrajectoryAction>

MANDATORY: TimeReference with Timing MUST be present — SR throws IndexError without it.
MANDATORY: Polyline MUST have exactly 4 or more vertices.
MANDATORY: Both hero AND adversary need separate AEB ManeuverGroups.
Lane change triggers when cartesianDistance between hero and adversary < 35.0m.

DO NOT use LaneChangeAction anywhere in this scenario.
=== END CCFhol SPECIAL RULES ===
"""

    if family.upper() in ("AEB", "VRU"):
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

            xosc2  = _sanitize_init_actions_and_patch_teleports(xosc, scenario)
            errors = validate_aeb_xosc(xosc2, scenario)

            if attempt == max_tries - 1 and errors:
                critical_errors = [e for e in errors if "missing required xml comment" not in e.lower()]
                if not critical_errors:
                    return xosc2.strip()
                last_errors = critical_errors
            elif not errors:
                return xosc2.strip()
            else:
                last_errors = errors

            prompt_for_attempt = _augment_user_prompt_with_errors(user_prompt, errors)

        raise RuntimeError(
            "XOSC generation failed after retries.\n"
            + "\n".join([f"- {e}" for e in last_errors])
        )

    # Non-AEB/VRU: one-shot
    result = call_llm_json(system_prompt, user_prompt, provider=provider)
    xosc = result.get("xosc")
    if isinstance(xosc, str) and xosc.strip():
        return xosc.strip()

    if enable_fallback_builder and _build_xosc_v5 is not None:
        return _build_xosc_v5(scenario)

    raise RuntimeError("LLM output did not contain a valid 'xosc' string and fallback is disabled.")
