"""
RAG2/prompts/python_prompts.py

IMPROVED VERSION:
- Better structured prompts for Claude
- Includes working code examples
- Clearer CARLA-specific guidance
- Step-by-step instructions
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


# =============================================================================
# SYSTEM PROMPTS (IMPROVED)
# =============================================================================

SYSTEM_PROMPT_AEB = """
You are an expert CARLA ScenarioRunner Python developer.

Your task: Generate a COMPLETE, RUNNABLE Python scenario module for CARLA ScenarioRunner.

OUTPUT FORMAT (CRITICAL):
- Return ONLY a single JSON object: {"carla_py": "<complete python code>"}
- NO markdown fences (no ```)
- NO explanations before or after the JSON
- NO additional text

CARLA SCENARIORUNNER REQUIREMENTS:
1. Must inherit from BasicScenario
2. Must implement these methods:
   - __init__
   - _initialize_actors
   - _create_behavior  
   - _create_test_criteria

3. Ego vehicle:
   - ALWAYS use self.ego_vehicles[0]
   - NEVER spawn ego manually

4. Target vehicle:
   - MUST spawn with world.try_spawn_actor()
   - MUST add to self.other_actors
   - MUST raise RuntimeError if spawn fails

5. Behavior tree:
   - Use py_trees.composites (Parallel, Sequence)
   - Import from srunner.scenariomanager.scenarioatomics.atomic_behaviors
   - Common behaviors: WaypointFollower, StopVehicle, ActorDestroy
   
6. Triggers:
   - Import from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions
   - Common: InTriggerDistanceToVehicle, InTimeToArrivalToVehicle

7. Criteria:
   - Import from srunner.scenariomanager.scenarioatomics.atomic_criteria
   - Common: CollisionTest

NULL HANDLING:
- Null/missing values are VALID
- Use safe defaults when needed:
  - ego_speed_kph: 50.0
  - target_speed_kph: 0.0 (for stationary scenarios)
  - initial_gap_m: 15.0
  - timeout_s: 60

CRITICAL:
- Include comment: # GENERATED_BY: Claude
- Code must be executable without modification
- No TODO placeholders
- No hardcoded file paths
""".strip()


SYSTEM_PROMPT_LSS = """
You are an expert CARLA ScenarioRunner developer for Lane Support System scenarios.

Follow same rules as AEB, but focus on:
- Lane-relative positioning
- Lateral offsets
- Lane change maneuvers

Output: {"carla_py": "<code>"}
No markdown. No explanations.
""".strip()


SYSTEM_PROMPT_VRU = """
You are an expert CARLA ScenarioRunner developer for VRU (pedestrian/cyclist) scenarios.

Follow same rules as AEB, but focus on:
- VRU actor spawning
- Crossing behavior
- Pedestrian/cyclist blueprints

Output: {"carla_py": "<code>"}
No markdown. No explanations.
""".strip()


# =============================================================================
# USER REQUIREMENTS (IMPROVED WITH EXAMPLES)
# =============================================================================

USER_REQUIREMENTS_AEB = """
SCENARIO FAMILY: AEB (Automatic Emergency Braking)

INPUT DATA STRUCTURE:
You will receive a scenario JSON with:
- user_config: Primary source of truth for parameters
- scenario_details: Fallback/reference data
- runtime_hints: Helper info (e.g., aeb_variant)

TRIGGER TYPE HANDLING:

1. START_IMMEDIATELY:
   - Scenario starts at simulation time > 0
   - NO TTC or distance trigger needed
   - Example:
     ```python
     # No trigger condition needed
     # Just start behaviors immediately
     root = py_trees.composites.Parallel(...)
     ```

2. TTC (Time To Collision):
   - Use InTimeToArrivalToVehicle
   - Get ttc_s from user_config.trigger.ttc_s (default: 2.0)
   - Example:
     ```python
     from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTimeToArrivalToVehicle
     
     trigger = InTimeToArrivalToVehicle(
         self._ego_vehicle,
         self._target_vehicle,
         time=self._ttc_s
     )
     ```

3. DISTANCE:
   - Use InTriggerDistanceToVehicle
   - Get distance_m from user_config.trigger.distance_m (default: 40.0)
   - Example:
     ```python
     from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle
     
     trigger = InTriggerDistanceToVehicle(
         self._ego_vehicle,
         self._target_vehicle,
         distance=self._trigger_distance
     )
     ```

PARAMETER EXTRACTION:
```python
# In __init__:
# Extract speeds (with defaults)
self._ego_speed_kph = config.get('ego_speed_kph', 50.0)
self._target_speed_kph = config.get('target_speed_kph', 0.0)

# Convert to m/s for CARLA
self._ego_speed = self._ego_speed_kph / 3.6
self._target_speed = self._target_speed_kph / 3.6

# Extract layout
self._initial_gap = config.get('initial_gap_m', 15.0)

# Extract trigger config
trigger_config = config.get('trigger', {})
self._trigger_type = trigger_config.get('type', 'START_IMMEDIATELY')
```

ACTOR SPAWNING PATTERN:
```python
def _initialize_actors(self, config):
    # Get ego (already spawned by ScenarioRunner)
    self._ego_vehicle = self.ego_vehicles[0]
    
    # Get ego's current location
    ego_location = CarlaDataProvider.get_location(self._ego_vehicle)
    ego_waypoint = self._map.get_waypoint(ego_location)
    
    # Calculate target spawn position (ahead of ego)
    target_waypoints = ego_waypoint.next(self._initial_gap)
    if not target_waypoints:
        raise RuntimeError("Could not find waypoint for target")
    
    target_waypoint = target_waypoints[0]
    target_transform = target_waypoint.transform
    
    # Spawn target
    blueprint = 'vehicle.lincoln.mkz2017'  # or from config
    target_bp = self._world.get_blueprint_library().find(blueprint)
    
    self._target_vehicle = self._world.try_spawn_actor(target_bp, target_transform)
    
    if self._target_vehicle is None:
        raise RuntimeError("Failed to spawn target vehicle")
    
    # Set initial velocity
    if self._target_speed == 0:
        # Stationary target
        self._target_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    
    # Register
    self.other_actors.append(self._target_vehicle)
```

BEHAVIOR TREE PATTERN (USE PARALLEL):
```python
def _create_behavior(self):
    # Ego behavior: drive at constant speed
    ego_drive = WaypointFollower(
        self._ego_vehicle,
        self._ego_speed,
        avoid_collision=False
    )
    
    # Target behavior: stay stopped or move
    if self._target_speed == 0:
        target_behavior = StopVehicle(self._target_vehicle, 1.0)
    else:
        target_behavior = WaypointFollower(
            self._target_vehicle,
            self._target_speed,
            avoid_collision=False
        )
    
    # Run in parallel
    parallel = py_trees.composites.Parallel(
        "Behaviors",
        policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
    )
    parallel.add_child(ego_drive)
    parallel.add_child(target_behavior)
    
    # Sequence with cleanup
    root = py_trees.composites.Sequence("MainSequence")
    root.add_child(parallel)
    root.add_child(ActorDestroy(self._target_vehicle))
    
    return root
```

CRITERIA PATTERN:
```python
def _create_test_criteria(self):
    criteria = []
    
    collision = CollisionTest(
        self._ego_vehicle,
        terminate_on_failure=False
    )
    criteria.append(collision)
    
    return criteria
```

REQUIRED IMPORTS:
```python
import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    StopVehicle,
    WaypointFollower
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    InTriggerDistanceToVehicle,
    InTimeToArrivalToVehicle
)
from srunner.scenarios.basic_scenario import BasicScenario
```

COMPLETE MINIMAL EXAMPLE:
```python
# GENERATED_BY: Claude

import carla
import py_trees

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy,
    StopVehicle,
    WaypointFollower
)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.basic_scenario import BasicScenario


class CCRsScenario(BasicScenario):
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, timeout=60):
        self.timeout = timeout
        self._world = world
        self._map = CarlaDataProvider.get_map()
        
        # Parameters
        self._ego_speed_kph = 50.0
        self._target_speed_kph = 0.0
        self._ego_speed = self._ego_speed_kph / 3.6
        self._target_speed = self._target_speed_kph / 3.6
        self._initial_gap = 15.0
        
        self._target_vehicle = None
        
        super(CCRsScenario, self).__init__(
            "CCRsScenario",
            ego_vehicles,
            config,
            world,
            debug_mode,
            terminate_on_failure=False
        )
    
    def _initialize_actors(self, config):
        self._ego_vehicle = self.ego_vehicles[0]
        
        ego_location = CarlaDataProvider.get_location(self._ego_vehicle)
        ego_waypoint = self._map.get_waypoint(ego_location)
        
        target_waypoints = ego_waypoint.next(self._initial_gap)
        if not target_waypoints:
            raise RuntimeError("No waypoint for target")
        
        target_transform = target_waypoints[0].transform
        
        blueprint = 'vehicle.lincoln.mkz2017'
        target_bp = self._world.get_blueprint_library().find(blueprint)
        
        self._target_vehicle = self._world.try_spawn_actor(target_bp, target_transform)
        
        if self._target_vehicle is None:
            raise RuntimeError("Target spawn failed")
        
        self._target_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.other_actors.append(self._target_vehicle)
    
    def _create_behavior(self):
        ego_drive = WaypointFollower(self._ego_vehicle, self._ego_speed, avoid_collision=False)
        target_stop = StopVehicle(self._target_vehicle, 1.0)
        
        parallel = py_trees.composites.Parallel(
            "Behaviors",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        parallel.add_child(ego_drive)
        parallel.add_child(target_stop)
        
        root = py_trees.composites.Sequence("Main")
        root.add_child(parallel)
        root.add_child(ActorDestroy(self._target_vehicle))
        
        return root
    
    def _create_test_criteria(self):
        return [CollisionTest(self._ego_vehicle, terminate_on_failure=False)]
    
    def __del__(self):
        self.remove_all_actors()
```

REMEMBER:
- Follow this structure exactly
- Use Parallel for concurrent behaviors
- Extract parameters from JSON in __init__
- Apply safe defaults for null values
- Include # GENERATED_BY: Claude
- Return only JSON: {"carla_py": "..."}
""".strip()


USER_REQUIREMENTS_LSS = """
SCENARIO FAMILY: LSS (Lane Support System)

Use same structure as AEB but focus on:
- Lane-relative positioning
- Lateral offsets for lane departure
- Lane change maneuvers if specified

Extract from user_config.lss:
- system: "LDW", "LKA", or "ELK"
- boundary_type: "road_edge", "solid_line", "dashed_line"
- departure_side: "left", "right", "both"
- ego_speed_kph
- selected_lateral_speed_mps

Safe defaults:
- ego_speed_kph: 70.0
- lateral_speed_mps: 0.4
- system: "LKA"
""".strip()


USER_REQUIREMENTS_VRU = """
SCENARIO FAMILY: VRU (Vulnerable Road User)

Use same structure as AEB but:
- Spawn VRU actor (pedestrian or cyclist)
- Implement crossing behavior
- Use WalkerWandering or similar for VRU motion

Extract from user_config.vru:
- vru_blueprint: pedestrian or bicycle blueprint
- crossing_side: "left" or "right"
- speed_mps: crossing speed
- path_offset_m: lateral start position

Safe defaults:
- speed_mps: 1.4 (walking speed)
- crossing_side: "right"
""".strip()


# =============================================================================
# PROMPT SELECTION
# =============================================================================

def pick_python_prompts(family: str) -> Tuple[str, str]:
    f = (family or "").strip().upper()
    if f == "LSS":
        return SYSTEM_PROMPT_LSS, USER_REQUIREMENTS_LSS
    if f == "VRU":
        return SYSTEM_PROMPT_VRU, USER_REQUIREMENTS_VRU
    return SYSTEM_PROMPT_AEB, USER_REQUIREMENTS_AEB


# =============================================================================
# PROMPT BUILDER (IMPROVED)
# =============================================================================

def build_python_prompts(
    *,
    scenario: Dict[str, Any],
    family: str,
    retrieved_context: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    IMPROVED: Better context formatting and clearer instructions
    """
    system_prompt, requirements = pick_python_prompts(family)

    # Format retrieved context more clearly
    context_lines = []
    if retrieved_context:
        context_lines.append("RETRIEVED EXAMPLES AND RULES:")
        context_lines.append("=" * 60)
        
        for i, hit in enumerate(retrieved_context[:5], 1):  # Limit to top 5
            meta = hit.get("metadata") or {}
            doc = (hit.get("document") or "").strip()
            
            if len(doc) > 1000:
                doc = doc[:1000] + "\n... (truncated)"
            
            context_lines.append(f"\n[CONTEXT {i}]")
            if meta:
                context_lines.append(f"Metadata: {json.dumps(meta, ensure_ascii=False)}")
            context_lines.append(doc)
            context_lines.append("-" * 40)
    
    context_blob = "\n".join(context_lines) if context_lines else "No additional context retrieved."

    # Format scenario JSON clearly
    scenario_blob = json.dumps(scenario, ensure_ascii=False, indent=2)

    # Build final user prompt
    user_prompt = f"""
{requirements}

{'='*60}
{context_blob}
{'='*60}

SCENARIO TO IMPLEMENT:
{scenario_blob}

{'='*60}

OUTPUT REQUIREMENTS:
1. Return ONLY this exact JSON structure:
   {{"carla_py": "<your complete python code here>"}}

2. NO markdown (no ``` or ```python)

3. NO explanations before or after the JSON

4. The code must:
   - Include # GENERATED_BY: Claude at the top
   - Be complete and executable
   - Follow the structure shown in examples above
   - Handle null values with safe defaults
   - Use Parallel for concurrent behaviors

5. Start your response with {{ and end with }}

GENERATE THE CODE NOW:
""".strip()

    return system_prompt, user_prompt