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

from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.state_representation import ProgressStateSE2, StateSE2
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import yaw_to_quaternion
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_policy import (
    IDMPolicy,
)
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_states import (
    IDMAgentState,
    IDMLeadAgentState,
)
from nuplan.planning.simulation.observation.smart_agents.idm_agents.utils import (
    box3d_to_polygon,
)
from nuplan.planning.simulation.path.interpolated_path import AbstractPath
from nuplan.planning.simulation.path.utils import get_trimmed_path_up_to_progress
from shapely.geometry import Polygon
from nuplan.planning.simulation.observation.smart_agents.idm_agents.idm_agent import (
    IDMAgent as IDMAgentParent,
)


class IDMAgent(IDMAgentParent):
    def __init__(
        self,
        start_iteration: int,
        initial_state: Box3D,
        path: AbstractPath,
        path_progress: float,
        policy: IDMPolicy,
    ):
        """
        Constructor for IDMAgent
        :param start_iteration: scenario iteration where agent first appeared
        :param initial_state: agent initial state
        :param path: agent initial state
        """

        self._start_iteration = (
            start_iteration  # scenario iteration where agent first appears
        )
        self._state: IDMAgentState = IDMAgentState(
            path_progress, initial_state.velocity[0]
        )
        self._initial_state = initial_state
        self._path: AbstractPath = path
        self._policy: IDMPolicy = policy
        self._size = initial_state.size

    def get_box_with_planned_trajectory(
        self, num_samples: int, sampling_time: float, agent_token: str
    ) -> Box3D:
        """
        Samples the the agent's trajectory. The velocity is assumed to be constant over the sampled trajectory
        :param num_samples: number of elements to sample.
        :param sampling_time: [s] time interval of sequence to sample from.
        :return: the agent's trajectory as a list of Box3D
        """
        return self._get_box_at_progress(
            self._get_bounded_progress(), num_samples, sampling_time, agent_token
        )

    def _get_box_at_progress(
        self,
        progress: float,
        num_samples: Optional[int] = None,
        sampling_time: Optional[float] = None,
        agent_token: Optional[str] = None,
    ) -> Box3D:
        """
        Returns the agent as a box at a given progress
        :param progress: the arc length along the agent's path
        :return: the agent as a Box3D object at the given progress
        """

        if self._path is not None:
            future_horizon_len_s = None
            future_interval_s = None
            future_centers = None
            future_orientations = None
            mode_probs = None

            progress = self._clamp_progress(progress)
            init_pose = self._path.get_state_at_progress(progress)
            init_orientation = yaw_to_quaternion(init_pose.heading)

            if num_samples is not None and sampling_time is not None:
                progress_samples = [
                    self._clamp_progress(
                        progress + self.velocity * sampling_time * (step + 1)
                    )
                    for step in range(num_samples)
                ]
                poses = [
                    self._path.get_state_at_progress(progress)
                    for progress in progress_samples
                ]
                future_horizon_len_s = num_samples * sampling_time
                future_interval_s = sampling_time
                future_centers = [[(pose.x, pose.y, 0.0) for pose in poses]]
                future_orientations = [
                    [yaw_to_quaternion(pose.heading) for pose in poses]
                ]
                mode_probs = [1.0]

            return Box3D(
                center=(init_pose.x, init_pose.y, 0),
                size=self._size,
                orientation=init_orientation,
                velocity=(
                    self._state.velocity * np.cos(init_pose.heading),
                    self._state.velocity * np.sin(init_pose.heading),
                    0,
                ),
                label=self._initial_state.label,
                future_horizon_len_s=future_horizon_len_s,
                future_interval_s=future_interval_s,
                future_centers=future_centers,
                future_orientations=future_orientations,
                mode_probs=mode_probs,
                track_token=agent_token,
            )
        return self._initial_state