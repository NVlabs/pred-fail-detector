from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
import numpy as np


def convert_box_to_node(box, prev_data):
    x_position = box.center[0]
    y_position = box.center[1]
    heading = box.orientation.yaw_pitch_roll[0]

    if len(prev_data) == 0:
        x_velocity = 0
        y_velocity = 0
    else:
        x_velocity = (x_position - prev_data[-1][0]) / dt
        y_velocity = (y_position - prev_data[-1][1]) / dt
    if len(prev_data) <= 1:
        x_acceleration = 0
        y_acceleration = 0
    else:
        x_acceleration = (x_velocity - prev_data[-1][2]) / dt
        y_acceleration = (y_velocity - prev_data[-1][3]) / dt

    data = [
        x_position,
        y_position,
        x_velocity,
        y_velocity,
        x_acceleration,
        y_acceleration,
        heading,
    ]

    return data


def convert_ego_state_to_node(ego_state):
    x_position = ego_state.center.x
    y_position = ego_state.center.y
    heading = ego_state.center.heading
    cc = np.cos(heading)
    ss = np.sin(heading)
    lon_vel = ego_state.dynamic_car_state.center_velocity_2d.x
    lat_vel = ego_state.dynamic_car_state.center_velocity_2d.y
    x_velocity = lon_vel * cc - lat_vel * ss
    y_velocity = lon_vel * ss + lat_vel * cc
    lon_acc = ego_state.dynamic_car_state.center_acceleration_2d.x
    lat_acc = ego_state.dynamic_car_state.center_acceleration_2d.y
    x_acceleration = lon_acc * cc - lat_acc * ss
    y_acceleration = lon_acc * ss + lat_acc * cc

    data = [
        x_position,
        y_position,
        x_velocity,
        y_velocity,
        x_acceleration,
        y_acceleration,
        heading,
    ]

    return data


def process_scenario(scenario):
    nodes = {}
    nodes["ego"] = {"data": []}
    T = scenario.get_number_of_iterations()
    for t in range(T):
        # Get all detections from the scenario at time step t
        ego_state = scenario.get_ego_state_at_iteration(t)
        detections = scenario.get_detections_at_iteration(t)

        # Process ego state
        ego_data = convert_ego_state_to_node(ego_state)
        nodes["ego"]["data"].append(ego_data)

        # Process all detections
        for box in detections.boxes:
            if 0 < box.label < 4:  # only add detections for: car, ped, bike
                if box.track_token not in nodes:
                    nodes[box.track_token] = {
                        "data": [],
                        "first_time_step": t,
                        "type": labelmap[box.label],
                    }
                elif (
                    nodes[box.track_token]["first_time_step"]
                    + len(nodes[box.track_token]["data"])
                    < t
                ):
                    assert (
                        False
                    ), "Not yet implemented: interpolate or make a new token for this node"

                data = convert_box_to_node(box, nodes[box.track_token]["data"])
                nodes[box.track_token]["data"].append(data)
    return nodes


from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import (
    convert_box3d_to_se2,
    create_path_from_se2,
)
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath


def get_agent_trajectory(
    scenario: NuPlanScenario, agent_track_token: str
) -> InterpolatedPath:
    path = []
    for t in range(scenario.get_number_of_iterations()):
        detections = scenario.get_detections_at_iteration(t)
        for box in detections.boxes:
            if box.track_token == agent_track_token:
                path.append(convert_box3d_to_se2(box))
    return create_path_from_se2(path)


if __name__ == "__main__":
    data_root = "/home/afarid/nuplan/dataset"
    version = "nuplan_v0.2_mini"
    max_scenarios = 10
    dt = 0.05  # maximum frequency: 20Hz
    labelmap = {
        0: "generic_object",
        1: "car",
        2: "ped",
        3: "bike",
        4: "traffic_cone",
        5: "barrier",
        6: "czone_sign",
    }

    # Sets up the filter for the database
    filter = ScenarioFilters(
        shuffle=False,
        flatten_scenarios=False,
        remove_invalid_goals=False,
        log_names=None,
        log_labels=None,
        max_scenarios_per_log=None,
        scenario_types=None,
        scenario_tokens=None,
        map_name=None,
        limit_scenarios_per_type=None,
        subsample_ratio=None,
        limit_total_scenarios=max_scenarios,
    )
    worker = RayDistributed()

    # Gathers the scenarios from the database
    scenario_builder = NuPlanScenarioBuilder(version, data_root)
    scenarios = scenario_builder.get_scenarios(filter, worker)

    scenes = {}
    for scenario in scenarios:
        print("Processing scenario:", scenario.scenario_name)
        nodes = process_scenario(scenario)

        scenes[scenario.scenario_name] = {"nodes": nodes}

    # Save the scenes here
