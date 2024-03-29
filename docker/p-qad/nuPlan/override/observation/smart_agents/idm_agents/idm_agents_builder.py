# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from audioop import avg
import logging
from typing import List, Optional, Tuple

import numpy as np
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import GraphEdgeMapObject
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import yaw_to_quaternion
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from override.observation.smart_agents.idm_agents.idm_agent import IDMAgent
from override.observation.smart_agents.idm_agents.idm_agent_manager import (
    UniqueIDMAgents,
)
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_policy import (
    IDMPolicy,
)
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import (
    box3d_to_polygon,
    convert_box3d_to_se2,
    create_path_from_se2,
    ego_state_to_box_3d,
)
from nuplan.planning.simulation.observation.smart_agents.occupancy_map.abstract_occupancy_map import (
    OccupancyMap,
)
from nuplan.planning.simulation.observation.smart_agents.occupancy_map.strtree_occupancy_map import (
    STRTreeOccupancyMapFactory,
)
from nuplan.planning.simulation.path.interpolated_path import InterpolatedPath
from tqdm import tqdm

logger = logging.getLogger(__name__)


def build_map_rails(
    agent: Box3D, map_api: AbstractMap, path_min_length: float = 200
) -> Tuple[Optional[InterpolatedPath], Optional[float], Optional[GraphEdgeMapObject]]:
    """
    Build a reference path for an agent.
    :param agent: The agent represented by a Box3D
    :param map_api: An AbstractMap instance
    :param path_min_length: [m] The minimum length of the path to be created
    :return: The constructed path as InterpolatedPath. If that path cannot be created then None.
    """
    agent_state = convert_box3d_to_se2(agent)
    if map_api.is_in_layer(agent_state, SemanticMapLayer.LANE):
        layer = SemanticMapLayer.LANE
    elif map_api.is_in_layer(agent_state, SemanticMapLayer.INTERSECTION):
        layer = SemanticMapLayer.LANE_CONNECTOR
    else:
        return None, None, None

    segments: List[GraphEdgeMapObject] = map_api.get_all_map_objects(agent_state, layer)
    if not segments:
        return None, None, None

    segment = segments[0]

    blp = segment.baseline_path().discrete_path()
    progress = segment.baseline_path().get_nearest_arc_length_from_position(agent_state)

    # Initialize with dummy path
    path = create_path_from_se2(blp)
    while path.get_end_progress() - progress < path_min_length:
        segments = segment.outgoing_edges()
        # Dead end found
        if not segments:
            break
        segment = segments[0]
        next_path = segment.baseline_path().discrete_path()

        blp += next_path
        path = create_path_from_se2(blp)

    return path, progress, segment


def build_idm_agents_on_map_rails(
    target_velocity: float,
    min_gap_to_lead_agent: float,
    headway_time: float,
    accel_max: float,
    decel_max: float,
    scenario: AbstractScenario,
) -> Tuple[UniqueIDMAgents, OccupancyMap]:
    """
    Build unique agents from a scenario. InterpolatedPaths are created for each agent according to their driven path

    :param target_velocity: Desired velocity in free traffic [m/s]
    :param min_gap_to_lead_agent: Minimum relative distance to lead vehicle [m]
    :param headway_time: Desired time headway. The minimum possible time to the vehicle in front [s]
    :param accel_max: maximum acceleration [m/s^2]
    :param decel_max: maximum deceleration (positive value) [m/s^2]
    :param scenario: scenario
    :return: a dictionary of IDM agent uniquely identified by an token
    """
    unique_agents: UniqueIDMAgents = {}
    first_states = {}

    include_updates = False

    detections = scenario.initial_detections
    detections2 = scenario.get_detections_at_iteration(1)
    map_api = scenario.map_api
    ego_box = ego_state_to_box_3d(scenario.get_ego_state_at_iteration(0))
    ego_box.token = "ego"
    agent_occupancy = STRTreeOccupancyMapFactory.get_from_boxes([ego_box])

    desc = "Converting detections to smart agents"

    for (
        agent_box
    ) in detections.boxes:  # tqdm(detections.boxes, desc=desc, leave=False):
        # filter for only vehicles
        if agent_box.label == 1 and agent_box.token not in first_states:

            path, progress, segment = build_map_rails(agent_box, map_api)
            # Ignore agents that a baseline path cannot be built for
            if path is None:
                continue

            # Snap agent to baseline path
            progress_state = path.get_state_at_progress(progress)
            agent_box.center = np.array([progress_state.x, progress_state.y, 0])
            agent_box.orientation = yaw_to_quaternion(progress_state.heading)

            # Check for collision
            if not agent_occupancy.intersects(box3d_to_polygon(agent_box)).is_empty():
                continue

            agent_occupancy.insert(agent_box.token, box3d_to_polygon(agent_box))

            first_states[agent_box.track_token] = (agent_box, path, progress, segment)

            if not include_updates:
                # Project velocity into local frame
                if np.isnan(agent_box.velocity).any():
                    ego_state = scenario.get_ego_state_at_iteration(0)
                    logger.debug(
                        f"Agents has nan velocity. Setting velocity to ego's velocity of "
                        f"{ego_state.dynamic_car_state.speed}"
                    )
                    agent_box.velocity = np.array(
                        [ego_state.dynamic_car_state.speed, 0.0, 0.0]
                    )
                else:
                    agent_box.velocity = (
                        np.hypot(agent_box.velocity[0], agent_box.velocity[1]),
                        0,
                        0,
                    )

                unique_agents[agent_box.token] = IDMAgent(
                    start_iteration=0,
                    initial_state=agent_box,
                    path=path,
                    path_progress=progress,
                    policy=IDMPolicy(
                        target_velocity,
                        min_gap_to_lead_agent,
                        headway_time,
                        accel_max,
                        decel_max,
                    ),
                )

    if include_updates:
        for agent_box2 in detections2.boxes:
            if agent_box2.track_token in first_states:
                agent_box, path, progress, segment = first_states[
                    agent_box2.track_token
                ]

                agent_state2 = convert_box3d_to_se2(agent_box2)
                progress2 = (
                    segment.baseline_path().get_nearest_arc_length_from_position(
                        agent_state2
                    )
                )
                progress_state2 = path.get_state_at_progress(progress2)
                center2 = np.array([progress_state2.x, progress_state2.y, 0])

                avg_velocity_over_scene = (
                    np.linalg.norm(center2 - agent_box.center) / 20
                )
                if avg_velocity_over_scene < 0.1:
                    target = 0.00001
                else:
                    target = target_velocity  # avg_velocity_over_scene*1.2

                agent_box.velocity = np.array([avg_velocity_over_scene, 0, 0])

                unique_agents[agent_box.token] = IDMAgent(
                    start_iteration=0,
                    initial_state=agent_box,
                    path=path,
                    path_progress=progress,
                    policy=IDMPolicy(
                        target,
                        min_gap_to_lead_agent,
                        headway_time,
                        accel_max,
                        decel_max,
                    ),
                )

    return unique_agents, agent_occupancy
