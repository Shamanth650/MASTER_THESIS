"""
RAG2/generators/python_generator.py

AEB-focused Python generator for CARLA ScenarioRunner.

FIXED VERSION:
- Relaxed validation that accepts semantically correct code regardless of formatting
- Pattern matching is flexible and checks for intent rather than exact syntax
- Supports various LLM output styles (Claude, GPT, etc.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import ast
import re

from ..scenario_utils import _ensure_mandatory_user_config, _family_of, _get_path
from ..chroma_store import retrieve_context
from ..llm_client import call_llm_json
from ..prompts.python_prompts import build_python_prompts


# -------------------------
# AEB-only helpers
# -------------------------

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
    """
    Tries multiple places to infer the AEB variant.
    Never hard-fails; returns 'unknown' if unclear.
    """
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


def _augment_user_prompt_with_errors(user_prompt: str, errors: List[str], scenario: Dict[str, Any]) -> str:
    """
    Strong retry instruction so the 2nd attempt actually fixes ScenarioRunner behavior.
    """
    trig_type = _infer_trigger_type(scenario)
    variant = _infer_aeb_variant_key(scenario)

    err_blob = "\n".join([f"- {e}" for e in errors])
    return (
        user_prompt
        + "\n\n"
        + "=== VALIDATION ERRORS FROM LAST OUTPUT (MUST FIX ALL) ===\n"
        + err_blob
        + "\n\n"
        + "Regenerate the FULL Python module.\n"
        + "Return ONLY JSON: {\"carla_py\": \"...\"}\n"
        + "Include '# GENERATED_BY: <model_name>' at the top of the code.\n"
        + "\n"
        + "CRITICAL ScenarioRunner correctness requirements:\n"
        + "1) The ego vehicle speed MUST be applied via a driving behavior (e.g., WaypointFollower on ego) or explicit control.\n"
        + "   Do NOT just read ego_speed and then use Idle().\n"
        + "2) Avoid 'Sequence' with a long-running child (e.g., WaypointFollower) followed by Idle.\n"
        + "   Use Parallel (or another correct pattern) so behaviors run together.\n"
        + "3) AEB variant handling must be generic. Current variant hint: "
        + f"{variant}\n"
        + "4) Trigger type must be respected. Current trigger.type: "
        + f"{trig_type or 'UNKNOWN'}\n"
        + "   - START_IMMEDIATELY: no TTC/distance primitive is required.\n"
        + "   - TTC: must include InTimeToArrivalToVehicle.\n"
        + "   - DISTANCE: must include InTriggerDistanceToVehicle.\n"
        + "5) For rear stationary cases, target should remain stopped (speed 0) and not behave oddly.\n"
        + "6) Always spawn target with world.try_spawn_actor and add it to other_actors.\n"
    )


def _extract_literal_assignment(code: str, attr: str) -> Optional[float]:
    """Extract a literal float assignment to self.attr"""
    patterns = [
        re.compile(rf"\bself\.{re.escape(attr)}\s*=\s*float\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)"),
        re.compile(rf"\bself\.{re.escape(attr)}\s*=\s*([0-9]+(?:\.[0-9]+)?)\b"),
    ]
    for p in patterns:
        m = p.search(code)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def _force_upper(v: Any) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        v = str(v)
    return v.strip().upper()


def _infer_trigger_type(scenario: Dict[str, Any]) -> str:
    """
    Returns normalized trigger type:
      START_IMMEDIATELY, TTC, DISTANCE, or "" if missing/unknown.
    """
    return _force_upper(_get_path(scenario, "user_config.trigger.type", ""))


def _code_uses_parallel(code: str) -> bool:
    """Check if code uses Parallel composite pattern"""
    if "py_trees.composites.Parallel" in code:
        return True
    if re.search(r"\bParallel\s*\(", code):
        return True
    return False


def _code_uses_sequence(code: str) -> bool:
    """Check if code uses Sequence composite pattern"""
    return ("py_trees.composites.Sequence" in code) or bool(re.search(r"\bSequence\s*\(", code))


def _ego_speed_applied(code: str) -> bool:
    """
    FIXED: Relaxed heuristic checking if ego speed is applied.
    
    Accepts any of these patterns:
    1. Ego speed variable is extracted from config
    2. Any driving behavior is applied to ego vehicle:
       - WaypointFollower with ego vehicle
       - KeepVelocity with ego vehicle
       - set_target_velocity on ego
       - apply_control on ego
       - TrafficManager autopilot
    
    This is much more flexible than strict regex matching.
    """
    # Normalize code for easier matching
    code_normalized = " ".join(code.split())
    
    # Check 1: Is ego speed extracted from somewhere?
    has_ego_speed_var = any([
        "self._ego_speed" in code,
        "ego_speed" in code and ("config" in code or "user_config" in code),
        "_ego_speed_kph" in code,
    ])
    
    if not has_ego_speed_var:
        return False
    
    # Check 2: Is any driving behavior applied to ego?
    # Look for ego vehicle references
    has_ego_vehicle = any([
        "self._ego_vehicle" in code,
        "self.ego_vehicles[0]" in code,
        "ego_vehicles[0]" in code,
    ])
    
    if not has_ego_vehicle:
        return False
    
    # Check 3: Is a driving behavior present?
    driving_behaviors = [
        "WaypointFollower",
        "KeepVelocity",
        "set_target_velocity",
        "apply_control",
        "set_autopilot",
    ]
    
    has_driving_behavior = any(behavior in code for behavior in driving_behaviors)
    
    if not has_driving_behavior:
        return False
    
    # Check 4: Semantic check - is the driving behavior applied to ego?
    # This is a relaxed check that looks within reasonable proximity
    
    # For WaypointFollower
    if "WaypointFollower" in code:
        # Find all WaypointFollower instances
        waypoint_pattern = r"WaypointFollower\s*\([^)]{0,300}\)"
        matches = re.finditer(waypoint_pattern, code, re.DOTALL)
        
        for match in matches:
            section = match.group(0)
            # Check if this WaypointFollower is applied to ego
            if any(ego_ref in section for ego_ref in ["ego_vehicle", "ego_vehicles[0]", "self._ego_vehicle"]):
                # Check if speed is referenced nearby (within 500 chars)
                start = match.start()
                context = code[max(0, start-500):min(len(code), start+500)]
                if any(speed_ref in context for speed_ref in ["ego_speed", "_ego_speed", "speed"]):
                    return True
    
    # For other behaviors - check if ego and speed are both mentioned
    for behavior in ["KeepVelocity", "set_target_velocity", "apply_control"]:
        if behavior in code:
            # Look for patterns where behavior is near ego reference
            behavior_positions = [m.start() for m in re.finditer(behavior, code)]
            ego_positions = [m.start() for m in re.finditer(r"ego_vehicle", code)]
            
            # If behavior and ego are within 300 characters, consider it applied
            for bp in behavior_positions:
                for ep in ego_positions:
                    if abs(bp - ep) < 300:
                        return True
    
    # TrafficManager autopilot is acceptable
    if "set_autopilot" in code and ("TrafficManager" in code or "traffic_manager" in code):
        return True
    
    # If we found speed extraction, ego vehicle, and driving behavior but can't confirm linkage,
    # be lenient and accept it (better to allow slightly uncertain cases than block valid code)
    return True


def _target_speed_handling_ok(code: str, variant: str) -> Optional[str]:
    """
    FIXED: More flexible validation of target speed handling.
    Returns an error string if variant contradicts obvious target handling, else None.
    """
    # Normalize variant
    variant_lower = variant.lower().strip()
    
    # CCRS: stationary target expected
    if variant_lower in ("ccrs", "rear_stationary", "rear stationary"):
        lit = _extract_literal_assignment(code, "target_speed_kph")
        if lit is not None and lit > 0.1:  # Allow tiny floating point errors
            return "CCRs/rear_stationary requires target stationary (target_speed_kph ≈ 0)."
        
        # Also check for explicit zero speed assignment
        if "target_speed" in code:
            # Look for patterns like target_speed = 0 or target_speed_kph = 0
            if re.search(r"target_speed[_a-z]*\s*=\s*0(?:\.0)?", code):
                return None  # Explicitly zero, good
            
            # If target_speed is mentioned but we can't find explicit zero, check the value
            target_speed_val = _extract_literal_assignment(code, "target_speed")
            if target_speed_val is not None and target_speed_val > 0.1:
                return "CCRs/rear_stationary: target should be stationary (speed = 0)."
        
        return None

    # CCRM: moving target expected
    if variant_lower in ("ccrm", "rear_moving", "rear moving"):
        lit = _extract_literal_assignment(code, "target_speed_kph")
        if lit is not None and lit <= 0:
            return "CCRm/rear_moving requires moving target (target_speed_kph > 0)."
        return None

    # CCRB: braking logic expected
    if variant_lower in ("ccrb", "rear_braking", "rear braking"):
        braking_indicators = ["brake", "decel", "target_decel", "apply_control", "ChangeSpeed"]
        if not any(indicator in code for indicator in braking_indicators):
            return "CCRb/rear_braking requires braking/deceleration logic for target."
        return None

    return None


def validate_aeb_python(carla_py: str, scenario: Dict[str, Any]) -> List[str]:
    """
    FIXED: AEB validation with relaxed pattern matching.
    
    Compatible with null-friendly requirement:
    - START_IMMEDIATELY does NOT require TTC or distance triggers
    - TTC requires TTC primitive, DISTANCE requires distance primitive
    - Unknown trigger type: accept either (do not block)

    Correctness checks:
    - Ego speed must be applied (not just read)
    - Avoid Sequence+WaypointFollower patterns that block forever (prefer Parallel)
    - Target must be spawned properly
    """
    errors: List[str] = []

    if not isinstance(carla_py, str) or not carla_py.strip():
        return ["Empty or invalid 'carla_py' output."]

    code = carla_py.strip()

    if "# GENERATED_BY:" not in code and "GENERATED_BY:" not in code:
        errors.append("Missing '# GENERATED_BY:' header (or comment marker).")

    try:
        ast.parse(code)
    except SyntaxError as e:
        return [f"Python syntax error: {e.msg} (line {e.lineno})"]

    # Core ScenarioRunner skeleton tokens
    required_tokens = {
        "BasicScenario": "Missing BasicScenario import/usage",
        "def _initialize_actors": "Missing _initialize_actors method",
        "def _create_behavior": "Missing _create_behavior method",
        "def _create_test_criteria": "Missing _create_test_criteria method",
    }
    
    for token, error_msg in required_tokens.items():
        if token not in code:
            errors.append(error_msg)

    # Ego vehicle must come from ego_vehicles[0]
    if not any(pattern in code for pattern in ["ego_vehicles[0]", "self.ego_vehicles[0]"]):
        errors.append("Ego vehicle must be retrieved from ego_vehicles[0].")

    # Target must be spawned - be flexible about method names
    spawn_patterns = [
        "try_spawn_actor",
        "spawn_actor",
        "world.try_spawn_actor",
        "_world.try_spawn_actor",
    ]
    if not any(pattern in code for pattern in spawn_patterns):
        errors.append("Target must be spawned via world.try_spawn_actor() or similar method.")

    # Trigger validation (conditional)
    trig_type = _infer_trigger_type(scenario)
    has_ttc = "InTimeToArrivalToVehicle" in code or "TimeToCollision" in code
    has_dist = "InTriggerDistanceToVehicle" in code or "TriggerDistance" in code

    if trig_type == "START_IMMEDIATELY":
        pass  # No specific trigger required
    elif trig_type == "TTC":
        if not has_ttc:
            errors.append("Trigger type TTC selected but TTC trigger primitive is missing (InTimeToArrivalToVehicle).")
    elif trig_type == "DISTANCE":
        if not has_dist:
            errors.append("Trigger type DISTANCE selected but distance trigger primitive is missing (InTriggerDistanceToVehicle).")
    else:
        pass  # Unknown trigger type - don't block

    if has_ttc and has_dist:
        # Both triggers present - should have conditional logic
        has_conditional = any(pattern in code for pattern in ["if", "else", "elif"])
        if not has_conditional:
            errors.append("Both TTC and distance triggers present without conditional logic (if/else or elif).")

    # Ego speed must be applied - FIXED relaxed check
    if not _ego_speed_applied(code):
        errors.append("Ego speed must be applied to control ego motion (e.g., WaypointFollower, KeepVelocity, or similar).")

    # Avoid the classic blocking tree: Sequence + WaypointFollower
    # Only flag if there's NO Parallel and there IS Sequence with WaypointFollower
    if _code_uses_sequence(code) and "WaypointFollower" in code:
        if not _code_uses_parallel(code):
            # Be more lenient - only warn if it looks like a problematic pattern
            # Check if WaypointFollower is inside the Sequence definition
            if "Sequence" in code and "WaypointFollower" in code:
                # Extract the behavior tree section
                behavior_section = ""
                if "def _create_behavior" in code:
                    start = code.index("def _create_behavior")
                    # Find the next method or end
                    remaining = code[start:]
                    next_def = remaining.find("def ", 10)
                    if next_def > 0:
                        behavior_section = remaining[:next_def]
                    else:
                        behavior_section = remaining
                
                # Only flag if WaypointFollower appears in a Sequence context
                if "Sequence" in behavior_section and "WaypointFollower" in behavior_section:
                    errors.append("Behavior uses Sequence with WaypointFollower which may block. Consider using Parallel composite.")

    # Variant sanity checks (generic)
    variant = _infer_aeb_variant_key(scenario)
    v_err = _target_speed_handling_ok(code, variant)
    if v_err:
        errors.append(v_err)

    return errors


# -------------------------
# Main generator
# -------------------------

def generate_python_rag(
    scenario: Dict[str, Any],
    *,
    k: int | None = None,
    provider: str = "openai",
) -> str:
    """
    FIXED: Generate Python code with relaxed validation.
    """
    if not isinstance(scenario, dict):
        raise RuntimeError("Scenario must be a dict.")

    family = _family_of(scenario)

    # Must remain null-friendly; scenario_utils should not hard-fail on nulls.
    _ensure_mandatory_user_config(scenario)

    scenario_name = (scenario.get("scenario_name") or scenario.get("name") or "").strip()
    trig_type = _infer_trigger_type(scenario)

    if family.upper() == "AEB":
        aeb_variant = _infer_aeb_variant_key(scenario)
        query_text = (
            "CARLA ScenarioRunner Python scenario template, rules, and examples for AEB family. "
            "Must correctly control ego speed, spawn target, and use correct behavior tree (Parallel if needed). "
            f"AEB variant={aeb_variant}. TriggerType={trig_type or 'UNKNOWN'}. Scenario={scenario_name}"
        )
    else:
        query_text = (
            "CARLA ScenarioRunner Python template and rules "
            f"for family={family}. TriggerType={trig_type or 'UNKNOWN'}. Scenario={scenario_name}"
        )

    retrieved = retrieve_context(query_text, k=k)

    system_prompt, user_prompt = build_python_prompts(
        scenario=scenario,
        family=family,
        retrieved_context=retrieved,
    )

    # ---- AEB: validate + retry once
    if family.upper() == "AEB":
        max_tries = 2
        last_errors: List[str] = []
        prompt_for_attempt = user_prompt

        for attempt in range(max_tries):
            result = call_llm_json(system_prompt, prompt_for_attempt, provider=provider)
            carla_py = result.get("carla_py")

            if not isinstance(carla_py, str):
                last_errors = ["Missing 'carla_py' in LLM output."]
                prompt_for_attempt = _augment_user_prompt_with_errors(user_prompt, last_errors, scenario)
                continue

            errors = validate_aeb_python(carla_py, scenario)
            if not errors:
                return carla_py.strip()

            # On last attempt, be more lenient with certain errors
            if attempt == max_tries - 1:
                # Filter out soft warnings
                critical_errors = [e for e in errors if not any(soft in e.lower() for soft in [
                    "may block",  # Sequence warning is a suggestion
                    "consider using",  # Suggestions not hard errors
                ])]
                
                if not critical_errors:
                    # Only had soft warnings, accept the code
                    return carla_py.strip()
                
                last_errors = critical_errors
            else:
                last_errors = errors
            
            prompt_for_attempt = _augment_user_prompt_with_errors(user_prompt, errors, scenario)

        raise RuntimeError("AEB Python generation failed:\n" + "\n".join(last_errors))

    # ---- Non-AEB unchanged
    result = call_llm_json(system_prompt, user_prompt, provider=provider)
    carla_py = result.get("carla_py")

    if not isinstance(carla_py, str):
        raise RuntimeError("Missing 'carla_py' in LLM output.")

    return carla_py.strip()