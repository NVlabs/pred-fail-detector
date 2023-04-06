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

from functools import cached_property
from override.observation.smart_agents.idm_agents.idm_agent_manager import (
    IDMAgentManager,
)
from override.observation.smart_agents.idm_agents.idm_agents_builder import (
    build_idm_agents_on_map_rails,
)
from nuplan.planning.simulation.observation.idm_agents_observation import (
    IDMAgentsObservation as IDMAgentsObservationParent,
)
from nuplan.planning.simulation.observation.observation_type import Detections


class IDMAgentsObservation(IDMAgentsObservationParent):
    @cached_property
    def _idm_agent_manager(self) -> IDMAgentManager:
        agents, agent_occupancy = build_idm_agents_on_map_rails(
            self._target_velocity,
            self._min_gap_to_lead_agent,
            self._headway_time,
            self._accel_max,
            self._decel_max,
            self._scenario,
        )

        return IDMAgentManager(agents, agent_occupancy)

    def get_observation(self) -> Detections:
        """Inherited, see superclass."""
        detections = self._idm_agent_manager.get_active_agents(
            self.current_iteration,
            self._planned_trajectory_samples,
            self._planned_trajectory_sample_interval,
        )
        return detections
