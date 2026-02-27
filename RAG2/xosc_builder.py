"""
RAG2/xosc_builder.py

Purpose:
- Deterministic fallback OpenSCENARIO (.xosc) generator.
- Used ONLY if LLM-based XOSC generation fails or is disabled.

This module:
- Does NOT use RAG
- Does NOT use LLMs
- Does NOT query Chroma
"""

from __future__ import annotations

from typing import Dict, Any
from xml.etree.ElementTree import Element, SubElement, tostring


def _build_xosc_v5(scenario: Dict[str, Any]) -> str:
    """
    Build a minimal valid OpenSCENARIO XML from scenario JSON.

    Input:
    - scenario: dict (from structured_scenarios.json)

    Output:
    - xosc_xml: str
    """

    user_config = scenario.get("user_config", {})
    scenario_name = scenario.get("scenario_name", "GeneratedScenario")

    # Root
    osc = Element("OpenSCENARIO")

    # -------------------------
    # FileHeader
    # -------------------------
    SubElement(
        osc,
        "FileHeader",
        {
            "revMajor": "1",
            "revMinor": "0",
            "date": "2026-01-01",
            "description": scenario_name,
            "author": "CLEAN_EURO_RAG",
        },
    )

    # -------------------------
    # Entities
    # -------------------------
    entities = SubElement(osc, "Entities")

    SubElement(
        entities,
        "ScenarioObject",
        {"name": "Ego"},
    ).append(
        SubElement(
            Element("Vehicle"),
            "Performance",
            {"maxSpeed": "50", "maxAcceleration": "3.0", "maxDeceleration": "8.0"},
        )
    )

    # -------------------------
    # Storyboard
    # -------------------------
    storyboard = SubElement(osc, "Storyboard")

    init = SubElement(storyboard, "Init")
    actions = SubElement(init, "Actions")

    private = SubElement(actions, "Private", {"entityRef": "Ego"})
    SubElement(private, "PrivateAction")  # placeholder init action

    story = SubElement(storyboard, "Story", {"name": "MainStory"})
    act = SubElement(story, "Act", {"name": "Act_1"})

    maneuver_group = SubElement(
        act, "ManeuverGroup", {"name": "ManeuverGroup_1", "maximumExecutionCount": "1"}
    )
    SubElement(maneuver_group, "Actors").append(
        SubElement(Element("EntityRef"), "EntityRef", {"entityRef": "Ego"})
    )

    maneuver = SubElement(maneuver_group, "Maneuver", {"name": "Maneuver_1"})
    event = SubElement(maneuver, "Event", {"name": "Event_1", "priority": "overwrite"})

    action = SubElement(event, "Action", {"name": "Action_1"})
    private_action = SubElement(action, "PrivateAction")

    longitudinal = SubElement(private_action, "LongitudinalAction")
    speed_action = SubElement(longitudinal, "SpeedAction")

    SubElement(
        speed_action,
        "SpeedActionTarget",
    ).append(
        SubElement(
            Element("AbsoluteTargetSpeed"),
            "AbsoluteTargetSpeed",
            {"value": str(user_config.get("ego_speed_kph", 30) / 3.6)},
        )
    )

    # StopTrigger
    stop_trigger = SubElement(act, "StopTrigger")
    SubElement(stop_trigger, "ConditionGroup")

    # Convert XML tree to string
    xml_bytes = tostring(osc, encoding="utf-8", method="xml")
    return xml_bytes.decode("utf-8")
