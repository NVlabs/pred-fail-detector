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

import time
from typing import Any, Dict, List, Tuple

import numpy as np
from bokeh.document.document import Document
from bokeh.layouts import column, row
from bokeh.models import Div, Select, Button
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from override.nuboard.simulation_tile import SimulationTile
from tornado.ioloop import IOLoop
from nuplan.planning.scenario_builder.abstract_scenario_builder import (
    AbstractScenarioBuilder,
)
import imageio as io
from bokeh.io.export import get_screenshot_as_png
from nuplan.planning.nuboard.tabs.scenario_tab import ScenarioTab as ScenarioTabParent


class ScenarioTab(ScenarioTabParent):
    def __init__(
        self,
        doc: Document,
        vehicle_parameters: VehicleParameters,
        scenario_builder: AbstractScenarioBuilder,
        file_paths: List[NuBoardFile],
        frame_rate,
    ):

        super().__init__(doc, vehicle_parameters, scenario_builder, file_paths)

        self._scalar_scenario_type_select = Select(title="Scenario type:")
        self._scalar_scenario_type_select.on_change(
            "value", self._scalar_scenario_type_select_on_change
        )

        self._scalar_scenario_name_select = Select(title="Scenario:")
        self._scalar_scenario_name_select.on_change(
            "value", self._scalar_scenario_name_select_on_change
        )

        self._save_button = Button(label="Save as gif")
        self._save_button.on_click(self._save_button_on_click)

        self._exit_button = Button(label="Exit")
        self._exit_button.on_click(self._exit_button_on_click)

        search_criteria = column(
            self.search_criteria_title,
            self._scalar_scenario_type_select,
            self._scalar_scenario_name_select,
            self._save_button,
            self._exit_button,
            height=self.search_criteria_height,
        )

        time_series_frame = column(
            Div(), sizing_mode="scale_height", height=self.plot_frame_sizes[1]
        )

        self._plot_layout = row([search_criteria, time_series_frame])
        self._frame_rate = frame_rate
        self.simulation_tile = SimulationTile(
            scenario_builder=self._scenario_builder,
            doc=self._doc,
            vehicle_parameters=vehicle_parameters,
        )

        self._init_selection()

    def _update_scenario_plot(self) -> None:
        """Update scenario plots when selection is made."""
        # Render simulations.
        simulation_layouts = self._render_simulations()

        col = column(
            simulation_layouts,
            sizing_mode="scale_height",
            height=self.plot_frame_sizes[1],
        )

        self._plot_layout.children[1] = col

    def _save_button_on_click(self, *args) -> None:
        scenario_name = self._scalar_scenario_name_select.value
        if scenario_name == "":
            print("nothing to save")
            return
        else:
            simulation_tile = self.simulation_tile

            num_imgs = simulation_tile._sliders[0].end + 1
            filenames = []
            figure_index = 0
            gifname = "plots/" + scenario_name[:6] + ".gif"

            with io.get_writer(gifname, mode="I", fps=self._frame_rate) as writer:
                for i in range(num_imgs):
                    print("saving frame", i, end="\r")
                    filename = "plotting/gif_imgs/" + str(i) + ".png"
                    filenames.append(filename)

                    simulation_tile._render_plots(
                        frame_index=i, figure_index=figure_index
                    )
                    plot = simulation_tile._figures[figure_index]
                    img = get_screenshot_as_png(plot)
                    writer.append_data(np.asarray(img))
                print()
            print("Saved as gif:", gifname)

    def _exit_button_on_click(self, *args) -> None:
        print("Exiting")
        IOLoop.current().stop()
