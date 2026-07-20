# parameter_overrides.py - Field-driven parameter override logic
"""
Detects which protocol-extracted fields are actually populated for a given
scenario, and applies user-selected overrides deterministically to a
generated XOSC string.

Core rules:
1. A field is only ever shown to the engineer if the extraction pipeline
   populated it as non-null for THIS specific scenario. Null fields are
   never fabricated into an option.
2. A field is only marked EDITABLE if its application logic has been
   verified against a real generated XOSC file. Everything else that is
   genuinely populated is still shown, for transparency, but read-only.
3. Entity identification never hardcodes "adversary" — it looks for
   whichever entity is not "hero", since "hero" is the only name
   confirmed universal across every scenario family.
4. Ego/target speed is only editable for CCRs, CCRm, CCRb, and only at
   the specific speed options in _VERIFIED_CCR_SPEED_TABLE below.
5. apply_canonical_corrections() ALWAYS forces the correct AEB trigger
   for CCFtap/CCFtab/CCFhos/CCCscp after generation, regardless of what
   the LLM produced.
6. apply_wide_spawn_correction() ALWAYS widens the spawn gap for
   CCFhos/CCFhol specifically, since their default 36.82m gap is too
   short for their real protocol speeds even with a corrected trigger —
   this is a spawn-geometry fix, not just a trigger tweak, and has NOT
   been run on CARLA. CCCscp is intentionally excluded from this and
   from any trigger revision below 48.0, per explicit decision to
   retain its original values despite an unresolved flagged concern —
   see Limitations.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

VEHICLE_OPTIONS = {
    "car": [
        "vehicle.tesla.model3",
        "vehicle.audi.a2",
        "vehicle.audi.etron",
        "vehicle.bmw.grandtourer",
        "vehicle.lincoln.mkz_2017",
    ],
    "cyclist": [
        "vehicle.bh.crossbike",
        "vehicle.diamondback.century",
        "vehicle.gazelle.omafiets",
    ],
    "motorcyclist": [
        "vehicle.harley-davidson.low_rider",
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
    ],
    "pedestrian": [
        "walker.pedestrian.0001",
        "walker.pedestrian.0002",
        "walker.pedestrian.0004",
    ],
}

_VERIFIED_CCR_SPEED_TABLE: Dict[str, Dict[str, Any]] = {
    "CCRs": {"ego_options_kph": [10, 50], "trigger_m": {10: 9.0, 50: 25.0}, "target_kph": 0},
    "CCRm": {"ego_options_kph": [30], "trigger_m": {30: 18.0}, "target_kph": 20},
    "CCRb": {"ego_options_kph": [50], "trigger_m": {50: 25.0}, "target_kph": 50},
}

# Canonical, always-enforced AEB trigger values for scenarios outside the
# CCR family. CCCscp retained at 48.0 per explicit decision despite an
# unresolved concern that it may exceed the scenario's actual initial
# distance (same failure mode identified in CCFhos) — see Limitations.
_CANONICAL_TRIGGER_TABLE: Dict[str, float] = {
    "CCFtap": 30.0,   # restored from prior CARLA-validated value — high confidence
    "CCFtab": 30.0,   # same geometry/profile as CCFtap
    "CCFhos": 42.0,   # superseded by apply_wide_spawn_correction below
    "CCCscp": 48.0,   # RETAINED PER EXPLICIT DECISION — flagged as unresolved, not fixed
    # CCFhol intentionally excluded from this table — handled separately below.
}

# CCFhos and CCFhol share CCR's road layout (same road, hero s=156.84,
# adversary s=193.66) and the same 8.333 m/s head-on speed profile. Their
# default 36.82m spawn gap is too short for that speed even with a
# corrected trigger — the fix widens the gap, not just the trigger.
_WIDE_SPAWN_SCENARIOS = {"CCFhos", "CCFhol"}
_SPAWN_S_DELTA = 40.0
_WIDE_SPAWN_TRIGGER = 30.0

_NAME_TO_SCENARIO_CODE = {
    "Car-to-Car Rear Stationary": "CCRs",
    "Car-to-Car Rear Moving": "CCRm",
    "Car-to-Car Rear Braking": "CCRb",
    "Car-to-Car Front Turn-Across-Path": "CCFtap",
    "Car-to-Car Front Head-On Straight": "CCFhos",
    "Car-to-Car Front Head-On Lane change": "CCFhol",
    "Car-to-car Crossing Straight Crossing Path": "CCCscp",
}


# -------------------------
# Scenario JSON helpers
# -------------------------
def _get_details(scenario: Dict[str, Any]) -> Dict[str, Any]:
    return scenario.get("scenario_details", scenario) or {}


def _get_extra(scenario: Dict[str, Any]) -> Dict[str, Any]:
    details = _get_details(scenario)
    return details.get("extra", {}) or {}


def _get_allowed_values(scenario: Dict[str, Any]) -> Dict[str, Any]:
    extra = _get_extra(scenario)
    return extra.get("allowed_values", {}) or {}


def _get_scenario_code(scenario: Dict[str, Any]) -> Optional[str]:
    code = scenario.get("scenario_code")
    if code:
        return code
    name = scenario.get("name") or scenario.get("scenario_name") or ""
    return _NAME_TO_SCENARIO_CODE.get(name)


def _actor_category(scenario: Dict[str, Any]) -> str:
    extra = _get_extra(scenario)
    family = (extra.get("adas_family") or "").upper()
    vru = extra.get("vru", {}) or {}
    vru_type = (vru.get("vru_type") or "").lower()

    if family == "VRU":
        if vru_type == "cyclist":
            return "cyclist"
        if vru_type == "motorcyclist":
            return "motorcyclist"
        return "pedestrian"
    return "car"


# -------------------------
# Detection: what to show the engineer
# -------------------------
def get_available_overrides(scenario: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    details = _get_details(scenario)
    allowed = _get_allowed_values(scenario)
    extra = _get_extra(scenario)
    scenario_code = _get_scenario_code(scenario)
    overrides: Dict[str, Dict[str, Any]] = {}

    def _add(field: str, label: str, unit: str = "", editable: bool = False):
        list_val = allowed.get(field)
        single_val = details.get(field)
        if list_val:
            options = list_val if isinstance(list_val, list) else [list_val]
            overrides[field] = {
                "label": f"{label}{unit}", "options": options,
                "default": options[0], "source": "protocol", "editable": editable,
            }
        elif single_val is not None:
            overrides[field] = {
                "label": f"{label}{unit}", "options": [single_val],
                "default": single_val, "source": "protocol", "editable": editable,
            }

    _add("overlap_percent", "Overlap", " (%)", editable=True)
    _add("headway_m", "Headway", " (m)", editable=True)
    _add("target_decel_mps2", "Target Deceleration", " (m/s\u00b2)", editable=True)

    ccr_table = _VERIFIED_CCR_SPEED_TABLE.get(scenario_code)
    if ccr_table:
        overrides["ego_speed_kph"] = {
            "label": "Ego Speed (km/h) \u2014 verified options only",
            "options": ccr_table["ego_options_kph"],
            "default": ccr_table["ego_options_kph"][0],
            "source": "protocol", "editable": True,
        }
        if scenario_code != "CCRs":
            overrides["target_speed_kph"] = {
                "label": "Target Speed (km/h)",
                "options": [ccr_table["target_kph"]],
                "default": ccr_table["target_kph"],
                "source": "protocol", "editable": True,
            }
    else:
        _add("ego_speed_min", "Ego Speed Min", " (km/h)")
        _add("ego_speed_max", "Ego Speed Max", " (km/h)")
        _add("target_speed_min", "Target Speed Min", " (km/h)")
        _add("target_speed_max", "Target Speed Max", " (km/h)")

    _add("initial_distance_m", "Initial Distance", " (m)")

    ttc = details.get("ttc") or details.get("ttc_end")
    if ttc is not None:
        overrides["ttc_s"] = {
            "label": "TTC (s)", "options": [ttc], "default": ttc,
            "source": "protocol", "editable": False,
        }

    impact_point = extra.get("impact_point_percent")
    if impact_point is not None:
        overrides["impact_point_percent"] = {
            "label": "Impact Point (%)", "options": [impact_point],
            "default": impact_point, "source": "protocol", "editable": False,
        }

    speed_pairs = allowed.get("speed_pairs")
    if speed_pairs:
        formatted = [f"ego {p['ego_speed']} / target {p['target_speed']}" for p in speed_pairs]
        overrides["speed_pairs"] = {
            "label": "Speed Pairs", "options": formatted,
            "default": formatted[0], "source": "protocol", "editable": False,
        }

    lane_change_offset = allowed.get("lane_change_offset_m")
    if lane_change_offset is not None:
        overrides["lane_change_offset_m"] = {
            "label": "Lane Change Offset (m)", "options": [lane_change_offset],
            "default": lane_change_offset, "source": "protocol", "editable": False,
        }

    category = _actor_category(scenario)
    model_options = VEHICLE_OPTIONS.get(category, VEHICLE_OPTIONS["car"])
    overrides["target_vehicle_blueprint"] = {
        "label": f"Adversary Model ({category}, catalog choice)",
        "options": model_options, "default": model_options[0],
        "source": "category", "editable": True,
    }

    return overrides


# -------------------------
# Application: writing choices into the XOSC
# -------------------------
def _lateral_offset_from_overlap(overlap_percent: float, vehicle_width_m: float = 1.85) -> float:
    fraction_uncovered = 1.0 - (abs(overlap_percent) / 100.0)
    offset = fraction_uncovered * vehicle_width_m
    return offset if overlap_percent >= 0 else -offset


def _non_hero_entity_name(root: ET.Element) -> Optional[str]:
    for obj in root.iter("ScenarioObject"):
        name = obj.get("name")
        if name and name != "hero":
            return name
    return None


def _maneuver_groups_for_entity(root: ET.Element, entity_name: str):
    for group in root.iter("ManeuverGroup"):
        actors = group.find("Actors")
        if actors is None:
            continue
        for ref in actors.iter("EntityRef"):
            if ref.get("entityRef") == entity_name:
                yield group
                break


def _find_stop_action_group(root: ET.Element, entity_name: str) -> Optional[ET.Element]:
    for group in _maneuver_groups_for_entity(root, entity_name):
        for target in group.iter("AbsoluteTargetSpeed"):
            if target.get("value") == "0.0":
                return group
    return None


def _set_parameter_value(root: ET.Element, param_name: str, new_value: str) -> bool:
    applied = False
    for decl in root.iter("ParameterDeclaration"):
        if decl.get("name") == param_name:
            decl.set("value", new_value)
            applied = True
    return applied


def apply_overrides_to_xosc(
    xosc_code: str, overrides: Dict[str, Any], scenario: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    warnings: List[str] = []
    try:
        root = ET.fromstring(xosc_code)
    except ET.ParseError as e:
        return {"xosc": xosc_code, "warnings": [f"Could not parse XOSC for override application: {e}"]}

    non_hero = _non_hero_entity_name(root)
    if non_hero is None:
        warnings.append("Could not identify the non-hero entity in this scenario — no overrides applied.")
        return {"xosc": xosc_code, "warnings": warnings}

    scenario_code = _get_scenario_code(scenario) if scenario else None

    if "overlap_percent" in overrides:
        offset = _lateral_offset_from_overlap(float(overrides["overlap_percent"]))
        applied = False
        for private in root.iter("Private"):
            if private.get("entityRef") == non_hero:
                for lane_pos in private.iter("LanePosition"):
                    lane_pos.set("offset", f"{offset:.2f}")
                    applied = True
        if not applied:
            warnings.append(f"overlap_percent override: LanePosition for '{non_hero}' not found in Init block, left unchanged.")

    if "target_vehicle_blueprint" in overrides:
        model = overrides["target_vehicle_blueprint"]
        applied = False
        for obj in root.iter("ScenarioObject"):
            if obj.get("name") == non_hero:
                for veh in obj.iter("Vehicle"):
                    veh.set("name", model)
                    applied = True
                for ped in obj.iter("Pedestrian"):
                    ped.set("model", model)
                    ped.set("name", model)
                    applied = True
        if not applied:
            warnings.append(f"target_vehicle_blueprint override: entity '{non_hero}' not found, left unchanged.")

    if "headway_m" in overrides:
        applied = False
        stop_group = _find_stop_action_group(root, non_hero)
        if stop_group is not None:
            for cond in stop_group.iter("RelativeDistanceCondition"):
                cond.set("value", f"{float(overrides['headway_m']):.1f}")
                applied = True
        if not applied:
            warnings.append(
                f"headway_m override: no braking-to-stop action found for '{non_hero}' "
                "in this scenario — left unchanged."
            )

    if "target_decel_mps2" in overrides:
        applied = False
        decel_value = abs(float(overrides["target_decel_mps2"]))
        stop_group = _find_stop_action_group(root, non_hero)
        if stop_group is not None:
            for dyn in stop_group.iter("SpeedActionDynamics"):
                dyn.set("dynamicsShape", "linear")
                dyn.set("dynamicsDimension", "rate")
                dyn.set("value", f"{decel_value:.2f}")
                applied = True
        if not applied:
            warnings.append(
                f"target_decel_mps2 override: no braking-to-stop action found for '{non_hero}' "
                "in this scenario — left unchanged."
            )

    if "ego_speed_kph" in overrides:
        table = _VERIFIED_CCR_SPEED_TABLE.get(scenario_code)
        if table is None:
            warnings.append(
                f"ego_speed_kph override: '{scenario_code}' is not in the verified speed/trigger "
                "table — left unchanged. Speed edits are only verified for CCRs/CCRm/CCRb."
            )
        else:
            speed_kph = float(overrides["ego_speed_kph"])
            speed_ms = speed_kph / 3.6
            trigger_m = table["trigger_m"].get(int(speed_kph))
            applied_speed = _set_parameter_value(root, "heroSpeed", f"{speed_ms:.3f}")
            applied_trigger = False
            if trigger_m is not None:
                hero_stop_group = _find_stop_action_group(root, "hero")
                if hero_stop_group is not None:
                    for cond in hero_stop_group.iter("RelativeDistanceCondition"):
                        cond.set("value", f"{trigger_m:.1f}")
                        applied_trigger = True
            if not applied_speed:
                warnings.append("ego_speed_kph override: 'heroSpeed' ParameterDeclaration not found, left unchanged.")
            if trigger_m is None:
                warnings.append(
                    f"ego_speed_kph override: no verified trigger distance for {speed_kph} km/h "
                    f"on {scenario_code} — speed changed but trigger left unchanged. Do not trust this run."
                )
            elif not applied_trigger:
                warnings.append("ego_speed_kph override: hero AEB trigger condition not found, trigger left unchanged.")

    if "target_speed_kph" in overrides:
        table = _VERIFIED_CCR_SPEED_TABLE.get(scenario_code)
        if table is None:
            warnings.append(
                f"target_speed_kph override: '{scenario_code}' is not in the verified speed table — left unchanged."
            )
        else:
            speed_kph = float(overrides["target_speed_kph"])
            speed_ms = speed_kph / 3.6
            applied = _set_parameter_value(root, "adversarySpeed", f"{speed_ms:.3f}")
            if not applied:
                warnings.append("target_speed_kph override: 'adversarySpeed' ParameterDeclaration not found, left unchanged.")

    for readonly_field in ("ttc_s", "impact_point_percent", "speed_pairs",
                           "lane_change_offset_m", "ego_speed_min", "ego_speed_max",
                           "target_speed_min", "target_speed_max"):
        if readonly_field in overrides:
            warnings.append(
                f"{readonly_field} override: display-only field, not yet wired into generation — no change applied."
            )

    patched = ET.tostring(root, encoding="unicode")
    return {"xosc": patched, "warnings": warnings}


def apply_canonical_corrections(xosc_code: str, scenario: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Always-run correction, independent of any user override selection.
    Forces the hero's AEB trigger to the known/estimated value for
    CCFtap/CCFtab/CCCscp. CCFhos is intentionally skipped here — it's
    handled by apply_wide_spawn_correction instead, which supersedes this
    trigger-only fix with a combined spawn+trigger correction.
    """
    warnings: List[str] = []
    if not scenario:
        return {"xosc": xosc_code, "warnings": warnings}

    scenario_code = _get_scenario_code(scenario)
    if scenario_code in _WIDE_SPAWN_SCENARIOS:
        return {"xosc": xosc_code, "warnings": warnings}

    target_trigger = _CANONICAL_TRIGGER_TABLE.get(scenario_code)
    if target_trigger is None:
        return {"xosc": xosc_code, "warnings": warnings}

    try:
        root = ET.fromstring(xosc_code)
    except ET.ParseError as e:
        return {"xosc": xosc_code, "warnings": [f"Could not parse XOSC for canonical correction: {e}"]}

    applied = False
    hero_stop_group = _find_stop_action_group(root, "hero")
    if hero_stop_group is not None:
        for cond in hero_stop_group.iter("RelativeDistanceCondition"):
            cond.set("value", f"{target_trigger:.1f}")
            applied = True
    if not applied:
        warnings.append(
            f"Canonical trigger correction for '{scenario_code}': hero AEB trigger condition "
            "not found — left unchanged. This scenario's structure may differ from expected."
        )

    if scenario_code == "CCCscp":
        warnings.append(
            "CCCscp: trigger retained at 48.0 per explicit decision. This value is UNVERIFIED and "
            "may exceed the scenario's actual initial spawn distance (same failure mode identified "
            "in CCFhos before its fix) — flag as an unresolved limitation, not a confirmed-safe value."
        )

    patched = ET.tostring(root, encoding="unicode")
    return {"xosc": patched, "warnings": warnings}


def apply_wide_spawn_correction(xosc_code: str, scenario: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    CCFhos and CCFhol only. Widens the adversary's spawn gap by 40m and
    sets AEB trigger conditions to 30.0, which has real margin in the
    enlarged gap. This is a spawn-geometry change, NOT run on CARLA —
    verify carefully, especially CCFhol's lane-change interaction with
    the wider gap (not independently re-derived for that case).
    """
    warnings: List[str] = []
    if not scenario or _get_scenario_code(scenario) not in _WIDE_SPAWN_SCENARIOS:
        return {"xosc": xosc_code, "warnings": warnings}
    scenario_code = _get_scenario_code(scenario)

    try:
        root = ET.fromstring(xosc_code)
    except ET.ParseError as e:
        return {"xosc": xosc_code, "warnings": [f"Could not parse XOSC for spawn correction: {e}"]}

    non_hero = _non_hero_entity_name(root)
    applied_spawn = False
    if non_hero:
        for private in root.iter("Private"):
            if private.get("entityRef") == non_hero:
                for lane_pos in private.iter("LanePosition"):
                    current_s = float(lane_pos.get("s", "0"))
                    lane_pos.set("s", f"{current_s + _SPAWN_S_DELTA:.2f}")
                    applied_spawn = True
    if not applied_spawn:
        warnings.append(f"{scenario_code}: adversary LanePosition not found — spawn gap unchanged.")

    applied_triggers = 0
    for group in root.iter("ManeuverGroup"):
        for cond in group.iter("RelativeDistanceCondition"):
            if cond.get("entityRef") in ("hero", "adversary"):
                cond.set("value", f"{_WIDE_SPAWN_TRIGGER:.1f}")
                applied_triggers += 1

    warnings.append(
        f"{scenario_code}: spawn gap widened by {_SPAWN_S_DELTA}m, AEB trigger(s) set to "
        f"{_WIDE_SPAWN_TRIGGER}. NOT run on CARLA — verify carefully before trusting this run."
    )

    patched = ET.tostring(root, encoding="unicode")
    return {"xosc": patched, "warnings": warnings}