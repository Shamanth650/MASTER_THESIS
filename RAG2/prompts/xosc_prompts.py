"""
RAG2/prompts/xosc_prompts.py
IMPROVED VERSION v5:
- Fixed placement rules: LanePosition for all entities (verified Town01 spawn points)
- Fixed init speed: always 0.0 in Init, $heroSpeed/$adversarySpeed ONLY in Story
- Fixed SimulationTimeCondition: always 1.0 not 0.1
- Fixed AEB trigger: longitudinal 12m for rear, cartesianDistance 30m for front
- Fixed spawn positions consistent across SYSTEM_PROMPT and working example
- Fixed default speeds: 8.0 m/s not 13.889
- Correct vehicle blueprints, map, date format, entity names
- AEB braking trigger included
- EnvironmentAction included
- Correct global StopTrigger criteria pattern
"""
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

# =============================================================================
# SYSTEM PROMPTS
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
- <RoadNetwork> with BOTH <LogicFile/> AND <SceneGraphFile filepath=""/>
- <Entities> (hero + adversary)
- <Storyboard>
  - <Init> with GlobalAction EnvironmentAction, TeleportAction and SpeedAction for each entity
  - <Story> → <Act> → <ManeuverGroup> → <Maneuver> → <Event> → <Action>
  - <StopTrigger>

CRITICAL PLACEMENT RULES:
- Every entity MUST have TeleportAction in Init
- Every TeleportAction MUST have Position
- ALWAYS use LanePosition for ALL entities — WorldPosition and RelativeRoadPosition cause off-road spawning in CARLA Town01
- NEVER use WorldPosition or RelativeRoadPosition

VERIFIED SPAWN POSITIONS FOR TOWN01 — USE EXACTLY THESE VALUES:

  CCRs (Car-to-Car Rear Stationary) — hero moves, adversary stationary ahead:
    Hero (moving, behind):     <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66" />
    Adversary (stationary, ahead): <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84" />

  CCRm (Car-to-Car Rear Moving) — both move, hero faster:
    Hero (behind, faster):     <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66" />
    Adversary (ahead, slower): <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84" />

  CCRb (Car-to-Car Rear Braking) — both move, adversary brakes:
    Hero (behind):             <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66" />
    Adversary (ahead):         <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84" />

  CCFtap (Car-to-Car Front Turn Across Path) — perpendicular roads:
    Hero:                      <LanePosition roadId="4" laneId="-1" offset="0.0" s="197.98" />
    Adversary:                 <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66" />

  CCFtab (Car-to-Car Front Turn Across Bicyclist) — perpendicular roads:
    Hero:                      <LanePosition roadId="4" laneId="-1" offset="0.0" s="197.98" />
    Bicyclist (adversary):     <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66" />

  CCFhos (Car-to-Car Front Head-On Straight) — opposite lanes:
    Hero:                      <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84" />
    Adversary (opposite lane): <LanePosition roadId="12" laneId="1" offset="0.0" s="193.66" />

  CCFhol (Car-to-Car Front Head-On Lane Change) — opposite lanes:
    Hero:                      <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84" />
    Adversary (opposite lane): <LanePosition roadId="12" laneId="1" offset="0.0" s="193.66" />

  CCFho (Car-to-Car Front Head-On) — opposite lanes:
    Hero:                      <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84" />
    Adversary (opposite lane): <LanePosition roadId="12" laneId="1" offset="0.0" s="193.66" />

SPEED SETTINGS — USE EXACTLY THESE VALUES PER SCENARIO:
  CCRs:  heroSpeed=8.0,  adversarySpeed=0.0  (adversary stationary)
  CCRm:  heroSpeed=8.0,  adversarySpeed=3.0  (adversary slower)
  CCRb:  heroSpeed=8.0,  adversarySpeed=8.0  (adversary brakes at t=5s)
  CCFtap: heroSpeed=10.0, adversarySpeed=3.0
  CCFtab: heroSpeed=10.0, adversarySpeed=3.0
  CCFhos: heroSpeed=8.0,  adversarySpeed=8.0
  CCFhol: heroSpeed=8.0,  adversarySpeed=8.0
  CCFho:  heroSpeed=8.0,  adversarySpeed=8.0

CRITICAL INIT SPEED RULES:
- Init AbsoluteTargetSpeed MUST always be value="0.0" for ALL entities
- DO NOT set initial speed to $heroSpeed or $adversarySpeed in Init
- Speed ramping MUST happen in Story ManeuverGroup using linear dynamics

CRITICAL STORY SPEED RULES:
- Story ManeuverGroup SpeedAction MUST use value="$heroSpeed" for hero
- Story ManeuverGroup SpeedAction MUST use value="$adversarySpeed" for adversary
- NEVER use 0.0 as the speed value in Story ManeuverGroup events (unless adversarySpeed=0.0)
- ALL SpeedAction in Story ManeuverGroup MUST use:
  dynamicsShape="linear" value="3.0" dynamicsDimension="time"
- DO NOT use dynamicsShape="step" in Story events

CRITICAL FORMAT RULES:
- date MUST always be "2020-03-20T12:00:00" — never just a date without time
- Map MUST always be Town01 with SceneGraphFile filepath=""
- maxAcceleration MUST always be 200 (not 10.0)
- SimulationTimeCondition value MUST always be "1.0" (never "0.1" or "0.0")
- Entity names MUST be "hero" and "adversary" (not "ego" and "target")
- hero vehicle MUST be "vehicle.lincoln.mkz_2017" (underscore before 2017)
- adversary vehicle MUST be "vehicle.tesla.model3"
- bicyclist vehicle MUST be "vehicle.bh.crossbike" with vehicleCategory="bicycle"
- Properties MUST include both type and role_name for each vehicle
- Always include EnvironmentAction block in Init GlobalAction
- Always include AEB braking trigger on hero
- Global StopTrigger MUST use criteria_CollisionTest ParameterCondition pattern

AEB TRIGGER RULES:
- Rear scenarios (CCRs, CCRm, CCRb):
  RelativeDistanceCondition entityRef="adversary" relativeDistanceType="longitudinal" value="12.0" freespace="true" rule="lessThan"
- Front and head-on scenarios (CCFtap, CCFtab, CCFhos, CCFhol, CCFho):
  RelativeDistanceCondition entityRef="adversary" relativeDistanceType="cartesianDistance" value="30.0" freespace="true" rule="lessThan"
- AEB Action MUST always be:
  SpeedAction to 0.0 m/s with dynamicsShape="linear" value="10.0" dynamicsDimension="distance"

NULL HANDLING:
- Null values are VALID
- Use safe defaults:
  - ego_speed_kph: use scenario-specific speed from SPEED SETTINGS above
  - target_speed_kph: use scenario-specific speed from SPEED SETTINGS above
  - initial_gap_m: 30.0
  - timeout_s: 60

REQUIRED COMMENT:
- Include: <!-- GENERATED_BY: Claude -->

NO TODO PLACEHOLDERS. Code must be executable.
""".strip()

SYSTEM_PROMPT_XOSC_LSS = SYSTEM_PROMPT_XOSC_BASE + "\n\nFocus on lane-relative positioning and lateral offsets for LSS scenarios."
SYSTEM_PROMPT_XOSC_VRU = SYSTEM_PROMPT_XOSC_BASE + "\n\nFocus on VRU (pedestrian/cyclist) crossing behavior."

# =============================================================================
# USER REQUIREMENTS
# =============================================================================
USER_REQUIREMENTS_XOSC_COMMON = """
DATA SOURCES:
- user_config: Primary source of truth
- scenario_details: Fallback/reference only
- runtime_hints: Helper info

ENTITY NAMING:
- Use: "hero" for ego vehicle, "adversary" for target vehicle
- EntityRef must match ScenarioObject name exactly
""".strip()

USER_REQUIREMENTS_XOSC_AEB = """
SCENARIO FAMILY: AEB (Automatic Emergency Braking)

ENTITIES:
- Define exactly 2 entities: "hero" and "adversary"
- Both must be Vehicle type
- hero: vehicle.lincoln.mkz_2017 (NOTE: underscore before 2017)
- adversary: vehicle.tesla.model3
- For bicyclist scenarios (CCFtab): adversary MUST be vehicle.bh.crossbike with vehicleCategory="bicycle"

PARAMETER EXTRACTION:
- ego_speed_kph from user_config.dynamics.ego_speed_kph
- target_speed_kph from user_config.dynamics.target_speed_kph
- initial_gap_m from user_config.layout.initial_gap_m (default: 30.0)
- timeout_s from user_config.termination.timeout_s (default: 60)
- If speeds are null, use scenario-specific defaults from SPEED SETTINGS in system prompt

SPEED CONVERSION:
- kph to m/s: divide by 3.6
- Default speeds if null: use SPEED SETTINGS table from system prompt (NOT 13.889)

TRIGGER TYPE HANDLING:
1. START_IMMEDIATELY (use value="1.0" not "0.1"):
   <StartTrigger>
     <ConditionGroup>
       <Condition name="StartCondition" delay="0" conditionEdge="rising">
         <ByValueCondition>
           <SimulationTimeCondition value="1.0" rule="greaterThan"/>
         </ByValueCondition>
       </Condition>
     </ConditionGroup>
   </StartTrigger>

2. AEB BRAKE TRIGGER (always include on hero ManeuverGroup):
   For rear scenarios (CCRs, CCRm, CCRb):
   <RelativeDistanceCondition entityRef="adversary" relativeDistanceType="longitudinal" value="12.0" freespace="true" rule="lessThan"/>
   For front/head-on scenarios (CCFtap, CCFtab, CCFhos, CCFhol, CCFho):
   <RelativeDistanceCondition entityRef="adversary" relativeDistanceType="cartesianDistance" value="30.0" freespace="true" rule="lessThan"/>

MINIMAL WORKING EXAMPLE — CCRs scenario (follow this structure exactly):
<?xml version="1.0" encoding="UTF-8"?>
<!-- GENERATED_BY: Claude -->
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="0" date="2020-03-20T12:00:00" description="AEB CCRs Test" author=""/>
  <ParameterDeclarations>
    <ParameterDeclaration name="heroSpeed" parameterType="double" value="8.0"/>
    <ParameterDeclaration name="adversarySpeed" parameterType="double" value="0.0"/>
  </ParameterDeclarations>
  <CatalogLocations/>
  <RoadNetwork>
    <LogicFile filepath="Town01"/>
    <SceneGraphFile filepath=""/>
  </RoadNetwork>
  <Entities>
    <ScenarioObject name="hero">
      <Vehicle name="vehicle.lincoln.mkz_2017" vehicleCategory="car">
        <ParameterDeclarations/>
        <Performance maxSpeed="69.444" maxAcceleration="200" maxDeceleration="10.0"/>
        <BoundingBox>
          <Center x="1.5" y="0.0" z="0.9"/>
          <Dimensions width="2.1" length="4.5" height="1.8"/>
        </BoundingBox>
        <Axles>
          <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8" positionX="3.1" positionZ="0.3"/>
          <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties>
          <Property name="type" value="ego_vehicle"/>
          <Property name="role_name" value="hero"/>
        </Properties>
      </Vehicle>
    </ScenarioObject>
    <ScenarioObject name="adversary">
      <Vehicle name="vehicle.tesla.model3" vehicleCategory="car">
        <ParameterDeclarations/>
        <Performance maxSpeed="69.444" maxAcceleration="200" maxDeceleration="10.0"/>
        <BoundingBox>
          <Center x="1.5" y="0.0" z="0.9"/>
          <Dimensions width="2.1" length="4.5" height="1.8"/>
        </BoundingBox>
        <Axles>
          <FrontAxle maxSteering="0.5" wheelDiameter="0.6" trackWidth="1.8" positionX="3.1" positionZ="0.3"/>
          <RearAxle maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties>
          <Property name="type" value="simulation"/>
          <Property name="role_name" value="adversary"/>
        </Properties>
      </Vehicle>
    </ScenarioObject>
  </Entities>
  <Storyboard>
    <Init>
      <Actions>
        <GlobalAction>
          <EnvironmentAction>
            <Environment name="Environment1">
              <TimeOfDay animation="false" dateTime="2020-03-20T12:00:00"/>
              <Weather cloudState="free">
                <Sun intensity="0.85" azimuth="0" elevation="1.31"/>
                <Fog visualRange="100000.0"/>
                <Precipitation precipitationType="dry" intensity="0.0"/>
              </Weather>
              <RoadCondition frictionScaleFactor="1.0"/>
            </Environment>
          </EnvironmentAction>
        </GlobalAction>
        <Private entityRef="hero">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>
              </Position>
            </TeleportAction>
          </PrivateAction>
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time"/>
                <SpeedActionTarget>
                  <AbsoluteTargetSpeed value="0.0"/>
                </SpeedActionTarget>
              </SpeedAction>
            </LongitudinalAction>
          </PrivateAction>
        </Private>
        <Private entityRef="adversary">
          <PrivateAction>
            <TeleportAction>
              <Position>
                <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
              </Position>
            </TeleportAction>
          </PrivateAction>
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time"/>
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
        <ManeuverGroup maximumExecutionCount="1" name="HeroManeuverGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="hero"/>
          </Actors>
          <Maneuver name="HeroManeuver">
            <Event name="HeroDrivesEvent" priority="overwrite">
              <Action name="HeroDrivesAction">
                <PrivateAction>
                  <LongitudinalAction>
                    <SpeedAction>
                      <SpeedActionDynamics dynamicsShape="linear" value="3.0" dynamicsDimension="time"/>
                      <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="$heroSpeed"/>
                      </SpeedActionTarget>
                    </SpeedAction>
                  </LongitudinalAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="StartCondition" delay="0" conditionEdge="rising">
                    <ByValueCondition>
                      <SimulationTimeCondition value="1.0" rule="greaterThan"/>
                    </ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
            <Event name="AEBBrakeEvent" priority="overwrite">
              <Action name="AEBBrakeAction">
                <PrivateAction>
                  <LongitudinalAction>
                    <SpeedAction>
                      <SpeedActionDynamics dynamicsShape="linear" value="10.0" dynamicsDimension="distance"/>
                      <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="0.0"/>
                      </SpeedActionTarget>
                    </SpeedAction>
                  </LongitudinalAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="AEBTrigger" delay="0" conditionEdge="rising">
                    <ByEntityCondition>
                      <TriggeringEntities triggeringEntitiesRule="any">
                        <EntityRef entityRef="hero"/>
                      </TriggeringEntities>
                      <EntityCondition>
                        <RelativeDistanceCondition entityRef="adversary" relativeDistanceType="longitudinal" value="12.0" freespace="true" rule="lessThan"/>
                      </EntityCondition>
                    </ByEntityCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        <ManeuverGroup maximumExecutionCount="1" name="AdversaryManeuverGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="adversary"/>
          </Actors>
          <Maneuver name="AdversaryManeuver">
            <Event name="AdversaryStaysEvent" priority="overwrite">
              <Action name="AdversaryStaysAction">
                <PrivateAction>
                  <LongitudinalAction>
                    <SpeedAction>
                      <SpeedActionDynamics dynamicsShape="linear" value="3.0" dynamicsDimension="time"/>
                      <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="$adversarySpeed"/>
                      </SpeedActionTarget>
                    </SpeedAction>
                  </LongitudinalAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="TargetStartCondition" delay="0" conditionEdge="rising">
                    <ByValueCondition>
                      <SimulationTimeCondition value="1.0" rule="greaterThan"/>
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
                <SimulationTimeCondition value="1.0" rule="greaterThan"/>
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
        <Condition name="criteria_CollisionTest" delay="0" conditionEdge="rising">
          <ByValueCondition>
            <ParameterCondition parameterRef="" value="" rule="lessThan"/>
          </ByValueCondition>
        </Condition>
        <Condition name="criteria_DrivenDistanceTest" delay="0" conditionEdge="rising">
          <ByValueCondition>
            <ParameterCondition parameterRef="distance_success" value="100" rule="lessThan"/>
          </ByValueCondition>
        </Condition>
      </ConditionGroup>
    </StopTrigger>
  </Storyboard>
</OpenSCENARIO>

KEY POINTS:
1. Follow the example structure exactly — do not deviate
2. ALWAYS use LanePosition for ALL entities — never WorldPosition or RelativeRoadPosition
3. Use EXACTLY the verified LanePosition values from VERIFIED SPAWN POSITIONS table above
4. For CCRs: hero s="193.66" (behind, moving), adversary s="156.84" (ahead, stationary)
5. For CCRm/CCRb: hero s="193.66" (behind), adversary s="156.84" (ahead)
6. For CCFtap/CCFtab: hero roadId=4 s="197.98", adversary roadId=12 s="193.66"
7. For CCFhos/CCFhol/CCFho: hero laneId=-1 s="156.84", adversary laneId=1 s="193.66"
8. Init AbsoluteTargetSpeed MUST ALWAYS be exactly value="0.0" — NEVER $heroSpeed or $adversarySpeed
9. Story ManeuverGroup SpeedAction MUST use value="$heroSpeed" for hero and value="$adversarySpeed" for adversary
10. Story SpeedAction MUST use dynamicsShape="linear" value="3.0" dynamicsDimension="time"
11. Entity names MUST be "hero" and "adversary"
12. date MUST be "2020-03-20T12:00:00" — never just a date
13. Map MUST be Town01 with SceneGraphFile filepath=""
14. maxAcceleration MUST be 200
15. SimulationTimeCondition MUST be 1.0 everywhere (never 0.1)
16. Always include EnvironmentAction in Init
17. Always include role_name property on both vehicles
18. Always include AEB braking trigger (longitudinal 12m for rear, cartesianDistance 30m for front)
19. Global StopTrigger MUST use criteria_CollisionTest pattern
20. Default speeds if null: use SPEED SETTINGS table — NEVER use 13.889
21. Return only JSON: {"xosc": "..."}
""".strip()

USER_REQUIREMENTS_XOSC_LSS = """
SCENARIO FAMILY: LSS (Lane Support System)
Follow same structure as AEB but:
- Use LanePosition for hero placement
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
# PROMPT BUILDER
# =============================================================================
def build_xosc_prompts(
    *,
    scenario: Dict[str, Any],
    family: str,
    retrieved_context: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Builds system + user prompts for xosc generation.
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
   - Follow the example structure above exactly
   - Handle null values with safe defaults from SPEED SETTINGS table
   - Include LanePosition for all entities using EXACTLY the verified Town01 spawn points
   - Include TeleportAction with LanePosition and SpeedAction (value=0.0) for all entities in Init
   - Include EnvironmentAction in Init
   - Include AEB braking trigger on hero
   - Use criteria_CollisionTest in global StopTrigger
   - Use $heroSpeed in Story HeroDrivesAction and $adversarySpeed in Story AdversaryDrivesAction
5. Start your response with {{ and end with }}

GENERATE THE XML NOW:
""".strip()

    return system_prompt, user_prompt
