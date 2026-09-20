"""
RAG2/prompts/xosc_prompts.py
Prompt definitions and prompt builder for OpenSCENARIO (XOSC) generation:
system prompts per scenario family (AEB, LSS, VRU), the matching user
requirement blocks, family-based prompt selection, and final prompt assembly.

------------------------------------------------------------
Responsible for: Defining the fixed Town01 spawn positions, default speeds
and AEB trigger rules given to the LLM, choosing the system/user prompt pair
for a scenario family, and combining the requirements, the retrieved
ChromaDB context and the scenario JSON into the final XOSC generation prompt.
Maintainer: shamanth.adiga@ltts.com
------------------------------------------------------------
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
- ALWAYS use LanePosition for ALL entities
- NEVER use WorldPosition or RelativeRoadPosition — both cause off-road/river spawns in CARLA Town01

VERIFIED SPAWN POSITIONS FOR TOWN01 — USE EXACTLY THESE VALUES:

  CCRs (Car-to-Car Rear Stationary):
    Hero (behind, moving):         <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Adversary (ahead, stationary): <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  CCRm (Car-to-Car Rear Moving):
    Hero (behind, faster):         <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Adversary (ahead, slower):     <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  CCRb (Car-to-Car Rear Braking):
    Hero (behind):                 <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Adversary (ahead, brakes):     <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  CCFtap (Car-to-Car Front Turn Across Path) — perpendicular roads:
    Hero:                          <LanePosition roadId="4"  laneId="-1" offset="0.0" s="197.98"/>
    Adversary:                     <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  CCFtab (Car-to-Car Front Turn Across Bicyclist) — perpendicular roads:
    Hero:                          <LanePosition roadId="4"  laneId="-1" offset="0.0" s="197.98"/>
    Bicyclist adversary:           <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  CCFhos (Car-to-Car Front Head-On Straight) — opposite lanes:
    Hero:                          <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Adversary (opposite lane):     <LanePosition roadId="12" laneId="1"  offset="0.0" s="193.66"/>

  CCFhol (Car-to-Car Front Head-On Lane Change) — opposite lanes:
    Hero:                          <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Adversary (starts opp. lane):  <LanePosition roadId="12" laneId="1"  offset="0.0" s="193.66"/>
    NOTE: adversary uses FollowTrajectoryAction to change lanes — see CCFhol SPECIAL RULES below

  CCCscp (Car-to-Car Crossing Straight Crossing Path) — perpendicular roads, same layout as CCFtap:
    Hero:                          <LanePosition roadId="4"  laneId="-1" offset="0.0" s="197.98"/>
    Adversary:                     <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

DEFAULT SPEED SETTINGS (m/s) — validated defaults, used whenever user_config supplies no usable value:
  CCRs:   heroSpeed=2.778,  adversarySpeed=0.0     (hero 10 km/h)
  CCRm:   heroSpeed=8.333,  adversarySpeed=5.556   (hero 30 km/h, target 20 km/h)
  CCRb:   heroSpeed=13.889, adversarySpeed=13.889  (both 50 km/h, fixed by the protocol)
  CCFtap: heroSpeed=10.0,   adversarySpeed=3.0
  CCFtab: heroSpeed=10.0,   adversarySpeed=3.0
  CCFhos: heroSpeed=8.333,  adversarySpeed=8.333
  CCFhol: heroSpeed=8.333,  adversarySpeed=8.333
  CCCscp: heroSpeed=10.0,   adversarySpeed=8.333

SPEED SOURCE RULES:
- Rear-approach scenarios (CCRs, CCRm, CCRb): use user_config.dynamics.ego_speed_kph and
  target_speed_kph (divide by 3.6) when non-null AND equal to a verified option below;
  otherwise use the default above. Verified options:
    CCRs: ego 10 or 50 km/h, target 0 km/h
    CCRm: ego 30 km/h only, target 20 km/h
    CCRb: ego 50 km/h only, target 50 km/h
- CCFtap, CCFtab, CCFhos, CCFhol, CCCscp: speeds are not user-editable; ALWAYS use the defaults above.

CRITICAL INIT SPEED RULES:
- Init AbsoluteTargetSpeed MUST always be value="0.0" for ALL entities
- DO NOT set initial speed to $heroSpeed or $adversarySpeed in Init
- Speed ramping MUST happen in Story ManeuverGroup

CRITICAL STORY SPEED RULES:
- Story SpeedAction for hero: value="$heroSpeed"
- Story SpeedAction for adversary: value="$adversarySpeed"
- ALL Story SpeedActions MUST use: dynamicsShape="linear" value="3.0" dynamicsDimension="time"
- DO NOT use dynamicsShape="step" in Story events

CRITICAL FORMAT RULES:
- date MUST always be "2020-03-20T12:00:00"
- Map MUST always be Town01 with SceneGraphFile filepath=""
- maxAcceleration MUST always be 200
- SimulationTimeCondition value MUST always be "1.0" (NEVER "0.1" or "0.0")
- Entity names MUST be "hero" and "adversary"
- hero vehicle MUST be "vehicle.lincoln.mkz_2017"
- adversary vehicle MUST be "vehicle.tesla.model3"
- bicyclist vehicle MUST be "vehicle.bh.crossbike" with vehicleCategory="bicycle"
- motorcycle vehicle MUST be "vehicle.kawasaki.ninja" with vehicleCategory="motorcycle"
- Properties MUST include both type and role_name for each vehicle
- Always include EnvironmentAction block in Init GlobalAction
- Always include AEB braking trigger on hero
- Global StopTrigger MUST use criteria_CollisionTest ParameterCondition pattern

AEB TRIGGER RULES:
Every hero AEB trigger is: RelativeDistanceCondition entityRef="adversary"
relativeDistanceType="cartesianDistance" value="<distance>" freespace="false" rule="lessThan"
- CCRs: distance = 1.5 x heroSpeed (m/s) + 4.0 m, verified values:
    9.0 at heroSpeed 2.778 (10 km/h);  25.0 at heroSpeed 13.889 (50 km/h)
- CCRm: distance = 18.0 (validated at heroSpeed 8.333; the CCRs formula does not apply
  because the lead vehicle keeps moving during the hero's brake)
- CCRb: hero AEB distance = 12.0 (plus the lead-vehicle brake group described below)
- CCFtap, CCFtab: distance = 30.0
- CCFhos: distance = 42.0
- CCFhol: distance = 20.0 for BOTH hero and adversary (each triggers on the other entity)
- CCCscp: distance = 48.0
- Front, head-on and crossing trigger distances are fixed validated values, not derived from user_config

AEB BRAKE ACTION (ALL scenarios):
- SpeedAction to 0.0 m/s
- dynamicsShape="linear" value="3.0" dynamicsDimension="time"

CCRb LEAD-VEHICLE BRAKING (CCRb ONLY):
- Add a separate ManeuverGroup named "AdversaryBrakeGroup" acting on the adversary,
  placed before the WaitGroup
- Action: SpeedAction to 0.0 m/s with dynamicsShape="linear"
- StartTrigger: ByEntityCondition, TriggeringEntities = hero, RelativeDistanceCondition
  entityRef="adversary" relativeDistanceType="cartesianDistance" value="<headway>"
  freespace="false" rule="lessThan"
- <headway> = scenario_details.headway_m when non-null (allowed values 12 or 40), otherwise 25.0
- Dynamics: if user_config.behavior.target_decel_mps2 is non-null, use dynamicsDimension="rate"
  with value = absolute value of target_decel_mps2 (allowed values 2 or 6);
  otherwise use dynamicsDimension="time" value="3.0"
- Sequence: the lead vehicle brakes first; the hero's own AEB trigger (12.0) engages afterwards
  in response to the closing distance

CCFhol SPECIAL RULES — LANE CHANGE:
- NEVER use LaneChangeAction — causes segmentation fault in CARLA 0.9.15
- Use FollowTrajectoryAction with Polyline vertices for adversary lane change
- FollowTrajectoryAction REQUIRES TimeReference child element:
  <TimeReference>
    <Timing domainAbsoluteRelative="absolute" scale="1.0" offset="0.0"/>
  </TimeReference>
- Polyline MUST have 4 or more vertices (fewer causes IndexError in SR)
- Lane change triggered when adversary is within 35m cartesianDistance of hero
- Both hero AND adversary need separate AEB ManeuverGroups
- CCFhol FollowTrajectoryAction example:
  <FollowTrajectoryAction>
    <Trajectory name="LaneChangePath" closed="false">
      <ParameterDeclarations/>
      <Shape>
        <Polyline>
          <Vertex time="0.0">
            <Position><LanePosition roadId="12" laneId="1"  s="193.66" offset="0.0"/></Position>
          </Vertex>
          <Vertex time="1.5">
            <Position><LanePosition roadId="12" laneId="1"  s="188.0"  offset="-1.75"/></Position>
          </Vertex>
          <Vertex time="3.0">
            <Position><LanePosition roadId="12" laneId="-1" s="182.0"  offset="0.0"/></Position>
          </Vertex>
          <Vertex time="10.0">
            <Position><LanePosition roadId="12" laneId="-1" s="170.0"  offset="0.0"/></Position>
          </Vertex>
        </Polyline>
      </Shape>
    </Trajectory>
    <TimeReference>
      <Timing domainAbsoluteRelative="absolute" scale="1.0" offset="0.0"/>
    </TimeReference>
    <TrajectoryFollowingMode followingMode="position"/>
  </FollowTrajectoryAction>

WAITGROUP RULES:
- ALWAYS add WaitGroup as the LAST ManeuverGroup in every Act
- WaitGroup keeps Act alive until 60s timeout
- Without WaitGroup, SR exits early when all ManeuverGroups complete
- WaitGroup triggers at SimulationTimeCondition value="55.0"
- WaitGroup example:
  <ManeuverGroup maximumExecutionCount="1" name="WaitGroup">
    <Actors selectTriggeringEntities="false">
      <EntityRef entityRef="adversary"/>
    </Actors>
    <Maneuver name="WaitManeuver">
      <Event name="WaitEvent" priority="overwrite">
        <Action name="WaitAction">
          <PrivateAction>
            <LongitudinalAction>
              <SpeedAction>
                <SpeedActionDynamics dynamicsShape="linear" value="1.0" dynamicsDimension="time"/>
                <SpeedActionTarget>
                  <AbsoluteTargetSpeed value="0.0"/>
                </SpeedActionTarget>
              </SpeedAction>
            </LongitudinalAction>
          </PrivateAction>
        </Action>
        <StartTrigger>
          <ConditionGroup>
            <Condition name="WaitTrigger" delay="0.0" conditionEdge="rising">
              <ByValueCondition>
                <SimulationTimeCondition value="55.0" rule="greaterThan"/>
              </ByValueCondition>
            </Condition>
          </ConditionGroup>
        </StartTrigger>
      </Event>
    </Maneuver>
  </ManeuverGroup>

GLOBAL STOPTRIGGER RULES:
- MUST use criteria_CollisionTest ParameterCondition
- parameterRef MUST be "criteria_CollisionTest" (not empty string)
- Without this SR exits immediately with "Nothing to analyze" error
- Do NOT include criteria_DrivenDistanceTest
- Correct pattern:
  <StopTrigger>
    <ConditionGroup>
      <Condition name="criteria_CollisionTest" delay="0.0" conditionEdge="rising">
        <ByValueCondition>
          <ParameterCondition parameterRef="criteria_CollisionTest" value="" rule="lessThan"/>
        </ByValueCondition>
      </Condition>
    </ConditionGroup>
  </StopTrigger>

NULL HANDLING:
- Use the validated defaults from the DEFAULT SPEED SETTINGS table above when a value is null (CCRb default: heroSpeed=13.889, adversarySpeed=13.889)
- initial_gap_m default: 30.0
- timeout_s default: 60

REQUIRED COMMENT:
- Include: <!-- GENERATED_BY: Claude -->

NO TODO PLACEHOLDERS. Code must be executable.
""".strip()

SYSTEM_PROMPT_XOSC_LSS = SYSTEM_PROMPT_XOSC_BASE + "\n\nFocus on lane-relative positioning and lateral offsets for LSS scenarios."
SYSTEM_PROMPT_XOSC_VRU = SYSTEM_PROMPT_XOSC_BASE + """

VRU-SPECIFIC RULES:

PEDESTRIAN ENTITY:
- Use <Pedestrian> tag NOT <Vehicle> tag
- model="walker.pedestrian.0001" mass="80.0" pedestrianCategory="pedestrian"
- Entity name: "pedestrian" (not "adversary")
- Properties: type="simulation" role_name="pedestrian"
- Pedestrian BoundingBox: Center x="0.0" y="0.0" z="0.9", Dimensions width="0.5" length="0.4" height="1.8"

CYCLIST ENTITY:
- Use <Vehicle> tag with name="vehicle.bh.crossbike" vehicleCategory="bicycle"
- Entity name: "adversary"

MOTORCYCLE ENTITY:
- Use <Vehicle> tag with name="vehicle.kawasaki.ninja" vehicleCategory="motorcycle"
- Entity name: "adversary"
- Same xosc structure as CCR/CCF scenarios

VRU SPAWN POSITIONS (Town01 validated):
  Pedestrian scenarios (CPFA-50, CPNA-25, CPNA-75, CPNCO-50, CPLA-25, CPLA-50):
    Hero:        <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Pedestrian:  <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  Cyclist scenarios (CBNA-50, CBNAO-50, CBFA-50, CBLA-25, CBLA-50):
    Hero:        <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Cyclist:     <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  Motorcyclist scenarios (CMRs, CMRb):
    Hero:        <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Motorcycle:  <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  CMFtap (Motorcyclist Front Turn Across Path):
    Hero:        <LanePosition roadId="4"  laneId="-1" offset="0.0" s="197.98"/>
    Motorcycle:  <LanePosition roadId="12" laneId="-1" offset="0.0" s="193.66"/>

  CMoncoming (Motorcyclist Oncoming):
    Hero:        <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Motorcycle:  <LanePosition roadId="12" laneId="1"  offset="0.0" s="193.66"/>

  CMovertaking (Motorcyclist Overtaking) — uses FollowTrajectoryAction:
    Hero:        <LanePosition roadId="12" laneId="-1" offset="0.0" s="156.84"/>
    Motorcycle:  <LanePosition roadId="12" laneId="1"  offset="0.0" s="193.66"/>

VRU SPEED SETTINGS:
VRU SPEED SETTINGS (ALL pedestrian speeds = 1.389 m/s = 5 km/h):
  CPFA-50:  heroSpeed=13.889, pedestrianSpeed=1.389
  CPNA-25:  heroSpeed=5.556,  pedestrianSpeed=1.389
  CPNA-75:  heroSpeed=11.111, pedestrianSpeed=1.389
  CPNCO-50: heroSpeed=8.333,  pedestrianSpeed=1.389
  CPLA-25:  heroSpeed=5.556,  pedestrianSpeed=1.389
  CPLA-50:  heroSpeed=8.333,  pedestrianSpeed=1.389
  CBNA-50:  heroSpeed=8.333,  adversarySpeed=4.167
  CBNAO-50: heroSpeed=8.333,  adversarySpeed=4.167
  CBFA-50:  heroSpeed=8.333,  adversarySpeed=4.167
  CBLA-25:  heroSpeed=5.556,  adversarySpeed=4.167
  CBLA-50:  heroSpeed=8.333,  adversarySpeed=4.167
  CMRs:     heroSpeed=8.333,  adversarySpeed=0.0
  CMRb:     heroSpeed=8.333,  adversarySpeed=8.333
  CMFtap:   heroSpeed=8.333,  adversarySpeed=8.333
  CMoncoming:   heroSpeed=8.333, adversarySpeed=8.333
  CMovertaking: heroSpeed=8.333, adversarySpeed=8.333

PEDESTRIAN MOTION RULES (CRITICAL):
- AcquirePositionAction and RoutingAction are BROKEN for pedestrians in SR 0.9.16
- NEVER use AcquirePositionAction or RoutingAction for pedestrians
- Pedestrian motion: SpeedAction ONLY
- Pedestrian walk trigger: RelativeDistanceCondition longitudinal value="40.0"
- AEB trigger for pedestrian scenarios: cartesianDistance value="20.0" freespace="false"
- AEB brake: dynamicsShape="linear" value="3.0" dynamicsDimension="time"

PEDESTRIAN CONTINUE GROUP:
- After hero brakes, pedestrian should continue walking
- Add PedestrianContinueGroup triggered when hero SpeedCondition value="1.0" rule="lessThan"

KNOWN SR 0.9.16 LIMITATIONS (document in xosc comments):
- CPNA and CPLA appear visually identical — pedestrian crossing not supported
- Pedestrian walks longitudinally as substitute for lateral crossing
- LaneChangeAction causes segfault — use FollowTrajectoryAction for CMovertaking
""".strip()

# =============================================================================
# USER REQUIREMENTS
# =============================================================================
USER_REQUIREMENTS_XOSC_COMMON = """
DATA SOURCES:
- user_config: Primary source of truth
- scenario_details: Fallback/reference only
- runtime_hints: Helper info

ENTITY NAMING:
- Use: "hero" for ego vehicle, "adversary" for target vehicle/cyclist/motorcycle
- For pedestrian scenarios: use "pedestrian" as entity name
- EntityRef must match ScenarioObject name exactly
""".strip()

USER_REQUIREMENTS_XOSC_AEB = """
SCENARIO FAMILY: AEB (Automatic Emergency Braking)

ENTITIES:
- Define exactly 2 entities: "hero" and "adversary"
- Both must be Vehicle type
- hero: vehicle.lincoln.mkz_2017
- adversary: vehicle.tesla.model3
- For CCFtab: adversary MUST be vehicle.bh.crossbike with vehicleCategory="bicycle"

PARAMETER EXTRACTION:
- ego_speed_kph from user_config.dynamics.ego_speed_kph
- target_speed_kph from user_config.dynamics.target_speed_kph
- initial_gap_m from user_config.layout.initial_gap_m (default: 30.0)
- timeout_s from user_config.termination.timeout_s (default: 60)
- headway_m from scenario_details.headway_m (CCRb only, see CCRb LEAD-VEHICLE BRAKING)
- target_decel_mps2 from user_config.behavior.target_decel_mps2 (CCRb only)
- If speeds are null or not a verified option, use the DEFAULT SPEED SETTINGS table from system prompt

SPEED CONVERSION:
- kph to m/s: divide by 3.6

TRIGGER:
- All triggers use SimulationTimeCondition value="1.0" rule="greaterThan"

CCFhol SPECIAL CASE:
- Adversary uses FollowTrajectoryAction (NOT LaneChangeAction)
- See CCFhol SPECIAL RULES in system prompt for exact XML

MINIMAL WORKING EXAMPLE — CCRs scenario:
<?xml version="1.0" encoding="UTF-8"?>
<!-- GENERATED_BY: Claude -->
<OpenSCENARIO>
  <FileHeader revMajor="1" revMinor="0" date="2020-03-20T12:00:00" description="AEB CCRs Test" author=""/>
  <ParameterDeclarations>
    <ParameterDeclaration name="heroSpeed"     parameterType="double" value="2.778"/>
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
          <RearAxle  maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties>
          <Property name="type"      value="ego_vehicle"/>
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
          <RearAxle  maxSteering="0.0" wheelDiameter="0.6" trackWidth="1.8" positionX="0.0" positionZ="0.3"/>
        </Axles>
        <Properties>
          <Property name="type"      value="simulation"/>
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
        <Private entityRef="adversary">
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
      </Actions>
    </Init>
    <Story name="AEBStory">
      <Act name="AEBTestAct">
        <ManeuverGroup maximumExecutionCount="1" name="HeroAccelGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="hero"/>
          </Actors>
          <Maneuver name="HeroAccelManeuver">
            <Event name="HeroAccelEvent" priority="overwrite">
              <Action name="HeroAccelAction">
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
                  <Condition name="HeroAccelStart" delay="0.0" conditionEdge="rising">
                    <ByValueCondition>
                      <SimulationTimeCondition value="1.0" rule="greaterThan"/>
                    </ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        <ManeuverGroup maximumExecutionCount="1" name="AdversaryAccelGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="adversary"/>
          </Actors>
          <Maneuver name="AdversaryAccelManeuver">
            <Event name="AdversaryAccelEvent" priority="overwrite">
              <Action name="AdversaryAccelAction">
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
                  <Condition name="AdversaryAccelStart" delay="0.0" conditionEdge="rising">
                    <ByValueCondition>
                      <SimulationTimeCondition value="1.0" rule="greaterThan"/>
                    </ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        <ManeuverGroup maximumExecutionCount="1" name="HeroAEBGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="hero"/>
          </Actors>
          <Maneuver name="HeroAEBManeuver">
            <Event name="HeroAEBEvent" priority="overwrite">
              <Action name="HeroAEBBrakeAction">
                <PrivateAction>
                  <LongitudinalAction>
                    <SpeedAction>
                      <SpeedActionDynamics dynamicsShape="linear" value="3.0" dynamicsDimension="time"/>
                      <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="0.0"/>
                      </SpeedActionTarget>
                    </SpeedAction>
                  </LongitudinalAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="HeroAEBTrigger" delay="0.0" conditionEdge="rising">
                    <ByEntityCondition>
                      <TriggeringEntities triggeringEntitiesRule="any">
                        <EntityRef entityRef="hero"/>
                      </TriggeringEntities>
                      <EntityCondition>
                        <RelativeDistanceCondition entityRef="adversary"
                          relativeDistanceType="cartesianDistance"
                          value="9.0" freespace="false" rule="lessThan"/>
                      </EntityCondition>
                    </ByEntityCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        <ManeuverGroup maximumExecutionCount="1" name="WaitGroup">
          <Actors selectTriggeringEntities="false">
            <EntityRef entityRef="adversary"/>
          </Actors>
          <Maneuver name="WaitManeuver">
            <Event name="WaitEvent" priority="overwrite">
              <Action name="WaitAction">
                <PrivateAction>
                  <LongitudinalAction>
                    <SpeedAction>
                      <SpeedActionDynamics dynamicsShape="linear" value="1.0" dynamicsDimension="time"/>
                      <SpeedActionTarget>
                        <AbsoluteTargetSpeed value="0.0"/>
                      </SpeedActionTarget>
                    </SpeedAction>
                  </LongitudinalAction>
                </PrivateAction>
              </Action>
              <StartTrigger>
                <ConditionGroup>
                  <Condition name="WaitTrigger" delay="0.0" conditionEdge="rising">
                    <ByValueCondition>
                      <SimulationTimeCondition value="55.0" rule="greaterThan"/>
                    </ByValueCondition>
                  </Condition>
                </ConditionGroup>
              </StartTrigger>
            </Event>
          </Maneuver>
        </ManeuverGroup>
        <StartTrigger>
          <ConditionGroup>
            <Condition name="ActStartCondition" delay="0.0" conditionEdge="rising">
              <ByValueCondition>
                <SimulationTimeCondition value="1.0" rule="greaterThan"/>
              </ByValueCondition>
            </Condition>
          </ConditionGroup>
        </StartTrigger>
        <StopTrigger>
          <ConditionGroup>
            <Condition name="ActTimeoutCondition" delay="0.0" conditionEdge="rising">
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
        <Condition name="criteria_CollisionTest" delay="0.0" conditionEdge="rising">
          <ByValueCondition>
            <ParameterCondition parameterRef="criteria_CollisionTest" value="" rule="lessThan"/>
          </ByValueCondition>
        </Condition>
      </ConditionGroup>
    </StopTrigger>
  </Storyboard>
</OpenSCENARIO>

KEY POINTS:
1. ALWAYS use LanePosition for ALL entities — never WorldPosition or RelativeRoadPosition
2. CCRs/CCRm/CCRb: hero s="156.84" (behind), adversary s="193.66" (ahead)
3. CCFtap/CCFtab/CCCscp: hero roadId=4 s="197.98", adversary roadId=12 s="193.66"
4. CCFhos/CCFhol: hero laneId=-1 s="156.84", adversary laneId=1 s="193.66"
5. Init AbsoluteTargetSpeed MUST ALWAYS be value="0.0"
6. Story SpeedAction: dynamicsShape="linear" value="3.0" dynamicsDimension="time"
7. SimulationTimeCondition MUST be 1.0 everywhere
8. AEB brake: dynamicsShape="linear" value="3.0" dynamicsDimension="time"
9. AEB trigger CCR (cartesianDistance, freespace="false"): CCRs 9.0 (2.778 m/s) or 25.0 (13.889 m/s), CCRm 18.0, CCRb hero 12.0 plus lead-vehicle brake at 25.0
10. AEB trigger (cartesianDistance, freespace="false"): CCFtap/CCFtab 30.0, CCFhos 42.0, CCFhol 20.0, CCCscp 48.0
11. CCFhol: use FollowTrajectoryAction NOT LaneChangeAction (segfault)
12. Always add WaitGroup at t=55s
13. Global StopTrigger: criteria_CollisionTest parameterRef="criteria_CollisionTest"
14. Do NOT include criteria_DrivenDistanceTest
15. Return only JSON: {"xosc": "..."}
""".strip()

USER_REQUIREMENTS_XOSC_VRU = """
SCENARIO FAMILY: VRU (Vulnerable Road User)

ENTITIES:
- hero: vehicle.lincoln.mkz_2017 (always)
- pedestrian scenarios: <Pedestrian> entity, model="walker.pedestrian.0001"
- cyclist scenarios: vehicle.bh.crossbike, vehicleCategory="bicycle"
- motorcyclist scenarios: vehicle.kawasaki.ninja, vehicleCategory="motorcycle"

PARAMETER EXTRACTION:
- heroSpeed from scenario VRU SPEED SETTINGS in system prompt
- adversarySpeed / pedestrianSpeed from VRU SPEED SETTINGS in system prompt

PEDESTRIAN MOTION (CRITICAL):
- AcquirePositionAction and RoutingAction are BROKEN in SR 0.9.16 for pedestrians
- Use SpeedAction ONLY for pedestrian movement
- Pedestrian walk trigger: longitudinal distance 40.0m from hero
- After hero stops (SpeedCondition < 1.0), add PedestrianContinueGroup to keep ped walking

AEB TRIGGER FOR VRU:
- cartesianDistance value="20.0" freespace="false" rule="lessThan"
- AEB brake: dynamicsShape="linear" value="3.0" dynamicsDimension="time"

ALWAYS include WaitGroup at t=55s and criteria_CollisionTest in global StopTrigger.
""".strip()

USER_REQUIREMENTS_XOSC_LSS = """
SCENARIO FAMILY: LSS (Lane Support System)
Follow same structure as AEB but:
- Use LanePosition for hero placement
- Include lateral offset parameters
- Define lane change maneuvers if needed
""".strip()

# =============================================================================
# PROMPT SELECTION
# =============================================================================
# Selects the system prompt and the matching user-requirements block for the
# given scenario family (AEB is the default; LSS and VRU have their own variants).
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
    return (
        SYSTEM_PROMPT_XOSC_BASE,
        USER_REQUIREMENTS_XOSC_COMMON + "\n\n" + USER_REQUIREMENTS_XOSC_AEB
    )

# =============================================================================
# PROMPT BUILDER
# =============================================================================
# Builds the final (system, user) prompt pair: the requirements block, up to five
# retrieved context documents (each cut to 1500 chars) and the full scenario JSON.
def build_xosc_prompts(
    *,
    scenario: Dict[str, Any],
    family: str,
    retrieved_context: List[Dict[str, Any]],
) -> Tuple[str, str]:
    system_prompt, requirements = pick_xosc_prompts(family)

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
   - Use LanePosition for ALL entities with EXACTLY the verified Town01 spawn points
   - Init speed = 0.0 for all entities
   - Story speed ramp: linear 3.0s to $heroSpeed / $adversarySpeed
   - SimulationTimeCondition = 1.0 everywhere
   - AEB brake: linear 3.0s to 0.0
   - WaitGroup at t=55s
   - criteria_CollisionTest in global StopTrigger
5. Start your response with {{ and end with }}

GENERATE THE XML NOW:
""".strip()

    return system_prompt, user_prompt
