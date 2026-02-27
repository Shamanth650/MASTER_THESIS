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


class CarToCarRearStationaryScenario(BasicScenario):
    """
    Car-to-Car Rear Stationary (CCRs) scenario.
    
    The ego vehicle approaches a stationary target vehicle from behind.
    This tests AEB (Automatic Emergency Braking) functionality.
    
    Based on Euro NCAP AEB CCRs test protocol:
    - AEB tested at 10-50 km/h
    - FCW tested at 55-80 km/h
    - Target vehicle is stationary (0 km/h)
    - Overlap range: -50% to +50% with 25% steps
    """
    
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, timeout=60):
        """
        Initialize the CCRs scenario.
        """
        self.timeout = timeout
        self._world = world
        self._map = CarlaDataProvider.get_map()
        
        # Extract parameters from config with safe defaults
        # Ego speed: default to 50 kph (within AEB test range 10-50 km/h)
        self._ego_speed_kph = 50.0
        self._target_speed_kph = 0.0  # Stationary target
        
        # Convert to m/s for CARLA
        self._ego_speed = self._ego_speed_kph / 3.6
        self._target_speed = self._target_speed_kph / 3.6
        
        # Layout parameters
        self._initial_gap = 50.0  # meters ahead of ego
        self._lateral_offset = 0.0  # meters (0% overlap by default)
        
        # Vehicle references
        self._ego_vehicle = None
        self._target_vehicle = None
        
        # Target blueprint
        self._target_blueprint = 'vehicle.lincoln.mkz2017'
        
        super(CarToCarRearStationaryScenario, self).__init__(
            "CarToCarRearStationaryScenario",
            ego_vehicles,
            config,
            world,
            debug_mode,
            terminate_on_failure=False
        )
    
    def _initialize_actors(self, config):
        """
        Initialize and spawn the target vehicle.
        Ego vehicle is already spawned by ScenarioRunner.
        """
        # Get ego vehicle (already spawned)
        self._ego_vehicle = self.ego_vehicles[0]
        
        # Get ego's current location and waypoint
        ego_location = CarlaDataProvider.get_location(self._ego_vehicle)
        ego_waypoint = self._map.get_waypoint(ego_location)
        
        # Calculate target spawn position (ahead of ego)
        target_waypoints = ego_waypoint.next(self._initial_gap)
        if not target_waypoints:
            raise RuntimeError("Could not find waypoint for target vehicle spawn")
        
        target_waypoint = target_waypoints[0]
        target_transform = target_waypoint.transform
        
        # Apply lateral offset if specified
        if self._lateral_offset != 0.0:
            # Get right vector for lateral offset
            forward_vec = target_transform.get_forward_vector()
            right_vec = carla.Vector3D(
                -forward_vec.y,
                forward_vec.x,
                0.0
            )
            target_transform.location.x += right_vec.x * self._lateral_offset
            target_transform.location.y += right_vec.y * self._lateral_offset
        
        # Spawn target vehicle
        blueprint_library = self._world.get_blueprint_library()
        target_bp = blueprint_library.find(self._target_blueprint)
        
        # Set a specific color for visibility
        if target_bp.has_attribute('color'):
            target_bp.set_attribute('color', '255,0,0')  # Red for visibility
        
        self._target_vehicle = self._world.try_spawn_actor(target_bp, target_transform)
        
        if self._target_vehicle is None:
            raise RuntimeError("Failed to spawn target vehicle at location: {}".format(
                target_transform.location))
        
        # Set target to stationary (zero velocity)
        self._target_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        
        # Apply handbrake to ensure it stays stationary
        control = carla.VehicleControl()
        control.hand_brake = True
        self._target_vehicle.apply_control(control)
        
        # Register the target vehicle
        self.other_actors.append(self._target_vehicle)
    
    def _create_behavior(self):
        """
        Create the behavior tree for the scenario.
        
        Ego vehicle drives forward at constant speed.
        Target vehicle remains stationary.
        """
        # Ego behavior: drive at constant speed towards target
        ego_drive = WaypointFollower(
            self._ego_vehicle,
            self._ego_speed,
            avoid_collision=False  # Let AEB system handle collision avoidance
        )
        
        # Target behavior: remain stationary
        target_stationary = StopVehicle(
            self._target_vehicle,
            brake_value=1.0  # Full brake to stay stopped
        )
        
        # Run behaviors in parallel
        parallel_behaviors = py_trees.composites.Parallel(
            "ParallelBehaviors",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        parallel_behaviors.add_child(ego_drive)
        parallel_behaviors.add_child(target_stationary)
        
        # Main sequence with cleanup
        root = py_trees.composites.Sequence("MainSequence")
        root.add_child(parallel_behaviors)
        root.add_child(ActorDestroy(self._target_vehicle))
        
        return root
    
    def _create_test_criteria(self):
        """
        Create test criteria for the scenario.
        
        Primary criterion: Collision detection
        """
        criteria = []
        
        # Collision test - detect if ego collides with target
        collision_criterion = CollisionTest(
            self._ego_vehicle,
            terminate_on_failure=False  # Continue to record collision data
        )
        criteria.append(collision_criterion)
        
        return criteria
    
    def __del__(self):
        """
        Cleanup when scenario is destroyed.
        """
        self.remove_all_actors()