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

from dataclasses import dataclass


@dataclass
class IDMAgentState:
    progress: float  # [m] distane a long a path
    velocity: float  # [m/s] velocity along the oath

    def to_array(self):
        return [self.progress, self.velocity]


@dataclass
class IDMLeadAgentState(IDMAgentState):
    length_rear: float  # [m] length from vehicle CoG to the rear bumper

    def to_array(self):
        return [self.progress, self.velocity, self.length_rear]
