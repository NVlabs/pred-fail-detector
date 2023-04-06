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

from bokeh.document.document import Document
from nuplan.planning.nuboard.utils.utils import read_nuboard_file_paths
from override.nuboard.scenario_tab import ScenarioTab
from nuplan.planning.nuboard.nuboard import NuBoard as NuBoardParent


class NuBoard(NuBoardParent):
    def __init__(self, *args, frame_rate=2, **kwargs):

        super().__init__(*args, **kwargs)
        self._frame_rate = frame_rate

    def main_page(self, doc: Document) -> None:
        self._doc = doc

        nuboard_files = read_nuboard_file_paths(file_paths=self._nuboard_paths)
        scenario_tab = ScenarioTab(
            file_paths=nuboard_files,
            scenario_builder=self._scenario_builder,
            doc=self._doc,
            vehicle_parameters=self._vehicle_parameters,
            frame_rate=self._frame_rate,
        )

        self._doc.add_root(scenario_tab._plot_layout)  # type: ignore
