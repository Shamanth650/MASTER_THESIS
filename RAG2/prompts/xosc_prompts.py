"""
RAG2/prompts/xosc_prompts.py

IMPROVED VERSION:
- Better structured for Claude
- Includes working OpenSCENARIO examples
- Clearer XML structure guidance
- Step-by-step instructions
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


# =============================================================================
# SYSTEM PROMPTS (IMPROVED)
# =============================================================================

SYSTEM_PROMPT_XOSC_BASE = """
You are an expert OpenSCENARIO 1.x XML author for CARLA/ScenarioRunner.

Your task: Generate a COMPLETE, VALID OpenSCENARIO XML document.

OUTPUT FORMAT (CRITICAL):
- Return ONLY a single JSON object: {"xosc": "<complete XML>"}
- NO markdown fences (no ```)
- NO explanations before or after the JSON
- NO additional text

XML REQUIREMENTS:
1. Must be well-formed XML
2. Must be valid OpenSCENARIO 1.x schema
3. Must be executable in CARLA ScenarioRunner

MANDATORY STRUCTURE:
- <?xml version="1.0" encoding="UTF-8"?>
- <OpenSCENARIO>
- <FileHeader>
- <ParameterDeclarations>
- <CatalogLocations/>
- <RoadNetwork>
- <Entities> (ego + target + optional vru)
- <Storyboard>
  - <Init> with placement (TeleportAction) and speeds
  - <Story> → <Act> → <ManeuverGroup> → <Maneuver> → <Event> → <Action>
  - <StopTrigger>

CRITICAL PLACEMENT RULES:
- Every entity MUST have TeleportAction in Init
- Every TeleportAction MUST have Position
- ALWAYS use LanePosition (roadId, laneId, s) — NEVER use WorldPosition with hardcoded x,y as they may be off-road
- ego: LanePosition roadId="0" laneId="-1" s="10.0"
- target: LanePosition roadId="0" laneId="-1" s=(10.0 + initial_gap_m)
- Every entity MUST have AbsoluteTargetSpeed in Init

NULL HANDLING:
- Null values are VALID
- Use safe defaults:
  - ego_speed_kph: 50.0 → 13.889 m/s
  - target_speed_kph: 0.0 → 0.0 m/s
  - initial_gap_m: 15.0
  - timeout_s: 60

REQUIRED COMMENT:
- Include: <!-- GENERATED_BY: Claude -->

NO TODO PLACEHOLDERS. Code must be executable.
""".strip()


SYSTEM_PROMPT_XOSC_LSS = SYSTEM_PROMPT_XOSC_BASE + "\n\nFocus on lane-relative positioning and lateral offsets for LSS scenarios."
SYSTEM_PROMPT_XOSC_VRU = SYSTEM_PROMPT_XOSC_BASE + "\n\nFocus on VRU (pedestrian/cyclist) crossing behavior."


# =============================================================================
# USER REQUIREMENTS (IMPROVED WITH EXAMPLES)
# =============================================================================

USER_REQUIREMENTS_XOSC_COMMON = """
DATA SOURCES:
- user_config: Primary source of truth
- scenario_details: Fallback/reference only
- runtime_hints: Helper info

ENTITY NAMING:
- Use: "ego", "target", "vru" (if applicable)
- EntityRef must match ScenarioObject name exactly
""".strip()


USER_REQUIREMENTS_XOSC_AEB = """
SCENARIO FAMILY: AEB (Automatic Emergency Braking)

ENTITIES:
- Define exactly 2 entities: "ego" and "target"
- Both must be Vehicle type
- Use appropriate CARLA blueprints

PARAMETER EXTRACTION:
- ego_speed_kph from user_config.dynamics.ego_speed_kph (default: 50.0)
- target_speed_kph from user_config.dynamics.target_speed_kph (default: 0.0)
- initial_gap_m from user_config.layout.initial_gap_m (default: 15.0)
- timeout_s from user_config.termination.timeout_s (default: 60)

SPEED CONVERSION:
- kph to m/s: divide by 3.6
- Example: 50 kph = 13.889 m/s

TRIGGER TYPE HANDLING:

1. START_IMMEDIATELY:
   ```xml
   <StartTrigger>
     <ConditionGroup>
       <Condition name="StartCondition" delay="0" conditionEdge="rising">
         <ByValueCondition>
           <SimulationTimeCondition value="0.0" rule="greaterThan"/>
         </ByValueCondition>
       </Condition>
     </ConditionGroup>
   </StartTrigger>
   ```

2. TTC (Time To Collision):
   ```xml
   <StartTrigger>
     <ConditionGroup>
       <Condition name="TTCCondition" delay="0" conditionEdge="rising">
         <ByEntityCondition>
           <TriggeringEntities triggeringEntitiesRule="any">
             <EntityRef entityRef="ego"/>
           </TriggeringEntities>
           <EntityCondition>
             <TimeToCollisionCondition alongRoute="true" freespace="false" rule="lessThan" value="2.0">
               <TimeToCollisionConditionTarget>
                 <EntityRef entityRef="target"/>
               </TimeToCollisionConditionTarget>
             </TimeToCollisionCondition>
           </EntityCondition>
         </ByEntityCondition>
       </Condition>
     </ConditionGroup>
   </StartTrigger>
   ```

3. DISTANCE:
   ```xml
   <StartTrigger>
     <ConditionGroup>
       <Condition name="DistanceCondition" delay="0" conditionEdge="rising">
         <ByEntityCondition>
           <TriggeringEntities triggeringEntitiesRule="any">
             <EntityRef entityRef="ego"/>
           </TriggeringEntities>
           <EntityCondition>
             <RelativeDistanceCondition entityRef="target" relativeDistanceType="longitudinal" rule="lessThan" value="40.0" freespace="false"/>
           </EntityCondition>
         </ByEntityCondition>
       </Condition>
     </ConditionGroup>
   </StartTrigger>
   ```

MINIMAL WORKING EXAMPLE:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- GENERATED_BY: Claude -->
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="0" date="2026-02-01" description="AEB CCRs Test" author="RAG System"/>
  
  <ParameterDeclarations>
    <ParameterDeclaration name="ego_speed" parameterType="double" value="13.889"/>
    <ParameterDeclaration name="target_speed" parameterType="double" value="0.0"/>
    <ParameterDeclaration name="initial_gap" parameterType="double" value="15.0"/>
  </ParameterDeclarations>
  
  <CatalogLocations/>
  
  <RoadNetwork>
    <LogicFile filepath="Town03"/>
  </RoadNetwork>
  
  <Entities>
    <ScenarioObject name="ego">
      <Vehicle name="vehicle.tesla.model3" vehicleCategory="car">
        <ParameterDeclarations/>
        <Performance maxSpeed="50.0" maxAcceleration="10.0" maxDeceleration="10.0"/>
        <BoundingBox>
          <Center x="1.5" y="0.0" z="0.9"/>
          <Dimensions width="2.1" length="4.5" height="1.8"/>
        </BoundingBox>
        <Axles>
          <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8" positionX="3.1" positionZ="0.3"/>
          <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties/>
      </Vehicle>
    </ScenarioObject>
    
    <ScenarioObject name="target">
      <Vehicle name="vehicle.lincoln.mkz2017" vehicleCategory="car">
        <ParameterDeclarations/>
        <Performance maxSpeed="50.0" maxAcceleration="10.0" maxDeceleration="10.0"/>
        <BoundingBox>
          <Center x="1.5" y="0.0" z="0.9"/>
          <Dimensions width="2.1" length="4.5" height="1.8"/>
        </BoundingBox>
        <Axles>
          <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8" positionX="3.1" positionZ="0.3"/>
          <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties/>
      </Vehicle>
    </ScenarioObject>
  </Entities>
  
  <Storyboard>
    <Init>
      <Actions>
        <Private entityRef="ego">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <LanePosition roadId="0" laneId="-1" offset="0.0" s="10.0"/>
              </Position>
            </TeleportAction>
          </PrivateAction>
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/>
                <SpeedActionTarget>
                  <AbsoluteTargetSpeed value="13.889"/>
                </SpeedActionTarget>
              </SpeedAction>
            </LongitudinalAction>
          </PrivateAction>
        </Private>
        
        <Private entityRef="target">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <LanePosition roadId="0" laneId="-1" offset="0.0" s="25.0"/>
              </Position>
            </TeleportAction>
          </PrivateAction>
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/>
                <SpeedActionTarget>
                  <AbsoluteTargetSpeed value="0.0"/>
                </SpeedActionTarget>
              </SpeedAction>
            </LongitudinalAction>
          </PrivateAction>
        </Private>
      </Actions>
    </Init>
    
    <Story name="AEBStory">
      <Act name="AEBTestAct">
        <ManeuverGroup maximumExecutionCount="1" name="EgoManeuverGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="ego"/>
          </Actors>
          <Maneuver name="EgoManeuver">
            <Event name="EgoDriveEvent" priority="overwrite">
              <Action name="EgoDriveAction">
                <PrivateAction>
                  <LongitudinalAction>
                    <SpeedAction>
                      <SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/>
                      <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="13.889"/>
                      </SpeedActionTarget>
                    </SpeedAction>
                  </LongitudinalAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="StartCondition" delay="0" conditionEdge="rising">
                    <ByValueCondition>
                      <SimulationTimeCondition value="0.0" rule="greaterThan"/>
                    </ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        
        <ManeuverGroup maximumExecutionCount="1" name="TargetManeuverGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="target"/>
          </Actors>
          <Maneuver name="TargetManeuver">
            <Event name="TargetStayStill" priority="overwrite">
              <Action name="TargetStayStillAction">
                <PrivateAction>
                  <LongitudinalAction>
                    <SpeedAction>
                      <SpeedActionDynamics dynamicsShape="step" value="0.0" dynamicsDimension="time"/>
                      <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="0.0"/>
                      </SpeedActionTarget>
                    </SpeedAction>
                  </LongitudinalAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="StartCondition" delay="0" conditionEdge="rising">
                    <ByValueCondition>
                      <SimulationTimeCondition value="0.0" rule="greaterThan"/>
                    </ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        
        <StartTrigger>
          <ConditionGroup>
            <Condition name="ActStart" delay="0" conditionEdge="rising">
              <ByValueCondition>
                <SimulationTimeCondition value="0" rule="greaterThan"/>
              </ByValueCondition>
            </Condition>
          </ConditionGroup>
        </StartTrigger>
        
        <StopTrigger>
          <ConditionGroup>
            <Condition name="Timeout" delay="0" conditionEdge="rising">
              <ByValueCondition>
                <SimulationTimeCondition value="60.0" rule="greaterThan"/>
              </ByValueCondition>
            </Condition>
          </ConditionGroup>
        </StopTrigger>
      </Act>
    </Story>
    
    <StopTrigger>
      <ConditionGroup>
        <Condition name="GlobalTimeout" delay="0" conditionEdge="rising">
          <ByValueCondition>
            <SimulationTimeCondition value="60.0" rule="greaterThan"/>
          </ByValueCondition>
        </Condition>
      </ConditionGroup>
    </StopTrigger>
  </Storyboard>
</OpenSCENARIO>
```

KEY POINTS:
1. Follow this structure exactly
2. ALWAYS use LanePosition — NEVER WorldPosition with hardcoded x,y coordinates
3. Convert kph to m/s (divide by 3.6)
4. Include <!-- GENERATED_BY: Claude -->
5. Every entity needs TeleportAction AND SpeedAction in Init
6. Return only JSON: {"xosc": "..."}
""".strip()


USER_REQUIREMENTS_XOSC_LSS = """
SCENARIO FAMILY: LSS (Lane Support System)

Follow same structure as AEB but:
- Use LanePosition instead of WorldPosition when appropriate
- Include lateral offset parameters
- Define lane change maneuvers if needed

Extract from user_config.lss:
- ego_speed_kph
- lateral_offset_m
- boundary_type
- system type (LDW/LKA/ELK)
""".strip()


USER_REQUIREMENTS_XOSC_VRU = """
SCENARIO FAMILY: VRU (Vulnerable Road User)

Follow same structure as AEB but:
- Add VRU entity (pedestrian or cyclist)
- Define crossing path
- Include VRU speed and position

Extract from user_config.vru:
- vru_blueprint
- crossing_side
- speed_mps
- path_offset_m
""".strip()


# =============================================================================
# PROMPT SELECTION
# =============================================================================

def pick_xosc_prompts(family: str) -> Tuple[str, str]:
    f = (family or "").strip().upper()
    
    if f == "LSS":
        return (
            SYSTEM_PROMPT_XOSC_LSS,
            USER_REQUIREMENTS_XOSC_COMMON + "\n\n" + USER_REQUIREMENTS_XOSC_LSS
        )
    
    if f == "VRU":
        return (
            SYSTEM_PROMPT_XOSC_VRU,
            USER_REQUIREMENTS_XOSC_COMMON + "\n\n" + USER_REQUIREMENTS_XOSC_VRU
        )
    
    # AEB (default)
    return (
        SYSTEM_PROMPT_XOSC_BASE,
        USER_REQUIREMENTS_XOSC_COMMON + "\n\n" + USER_REQUIREMENTS_XOSC_AEB
    )


# =============================================================================
# PROMPT BUILDER (IMPROVED)
# =============================================================================

def build_xosc_prompts(
    *,
    scenario: Dict[str, Any],
    family: str,
    retrieved_context: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    IMPROVED: Better formatting and clearer instructions
    """
    system_prompt, requirements = pick_xosc_prompts(family)

    # Format retrieved context
    context_lines = []
    if retrieved_context:
        context_lines.append("RETRIEVED EXAMPLES AND TEMPLATES:")
        context_lines.append("=" * 60)
        
        for i, hit in enumerate(retrieved_context[:5], 1):
            meta = hit.get("metadata") or {}
            doc = (hit.get("document") or "").strip()
            
            if len(doc) > 1500:
                doc = doc[:1500] + "\n... (truncated)"
            
            context_lines.append(f"\n[CONTEXT {i}]")
            if meta:
                context_lines.append(f"Metadata: {json.dumps(meta, ensure_ascii=False)}")
            context_lines.append(doc)
            context_lines.append("-" * 40)
    
    context_blob = "\n".join(context_lines) if context_lines else "No additional context retrieved."

    scenario_blob = json.dumps(scenario, ensure_ascii=False, indent=2)

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
   {{"xosc": "<your complete XML here>"}}

2. NO markdown (no ``` or ```xml)

3. NO explanations before or after the JSON

4. The XML must:
   - Include <?xml version="1.0" encoding="UTF-8"?>
   - Include <!-- GENERATED_BY: Claude -->
   - Be complete and valid OpenSCENARIO 1.x
   - Follow the structure in examples above
   - Handle null values with safe defaults
   - Include TeleportAction for all entities
   - Include SpeedAction for all entities

5. Start your response with {{ and end with }}

GENERATE THE XML NOW:
""".strip()

    return system_prompt, user_prompt
