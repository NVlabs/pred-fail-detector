import sys
from typing import Optional, Type
import numpy as np
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import (
    Detections,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.simulation_manager.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.interpolated import InterpolatedTrajectory
from primitives import Primitive_Generator
from nuplan_trajectron_utils import (
    convert_primitives_to_nodes,
    build_trajectron_env,
    add_observations_to_scene,
    load_trajectron_model,
    serialize_ego_trajectory,
)
from plot import new_plot
from labels import labels_1 as scene_labels

sys.path.append("../")
from detection.cost_util import CostUtil
from detection.anomaly_detector import (
    anomaly_detection_all,
    aggregate_anomaly_detection,
    aggregate_adaptive_planning,
)
from detection.query import Query
from detection.query_utils import query_node


class PredictorPlanner(AbstractPlanner):
    """
    Implements abstract planner interface.
    Used for simulating any ML planner trained through the nuPlan training framework.
    """

    def __init__(
        self, subsample_ratio, scenario, verbose=True, following_expert=False, exp=0
    ) -> None:
        eval_env = build_trajectron_env()
        self._stg = load_trajectron_model(eval_env)
        self._scene = eval_env.scenes[0]

        self._max_data_frequency = 20
        self._subsample_ratio = subsample_ratio
        self._future_horizon = 6.0  # model.future_trajectory_sampling.time_horizon
        self._total_num_steps = 40
        self._dt = 1 / (subsample_ratio * self._max_data_frequency)
        self._ph = 4
        self.num_predict_samples = 100

        self._expert_goal_state: Optional[StateSE2] = None
        self._mission_goal: Optional[StateSE2] = None
        self._map_name: Optional[str] = None
        self._map_api: Optional[AbstractMap] = None

        self.criterion = CostUtil(self._scene)
        self.mp_generator = Primitive_Generator(self._dt, self._ph, scenario)

        self._steps = -1
        self._tau = -1
        self._last_prediction_time = -1
        self._plan = None
        self._predictions = {}
        self._predicted_costs = {}
        self._achieved_costs = {}
        self._scene_radius = 100
        self.max_anomaly_rank = 0.0
        self._scenario = scenario
        self._expert_traj = serialize_ego_trajectory(scenario, subsample_ratio)
        self._verbose = verbose
        self._fast = True
        self.following_expert = following_expert
        self.evaluation = False
        self.advalues = []
        self.exp = exp

        if self.exp == 4:
            self._ph = 20

    def initialize(
        self,
        expert_goal_state: StateSE2,
        mission_goal: StateSE2,
        map_name: str,
        map_api: AbstractMap,
    ) -> None:
        """Inherited, see superclass."""
        self._expert_goal_state = expert_goal_state
        self._mission_goal = mission_goal
        self._map_name = map_name
        self._map_api = map_api

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return Detections  # type: ignore

    def compute_trajectory(
        self, iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> InterpolatedTrajectory:
        ### STEP 0: Setup
        # Setup initial scene processing from data in buffer
        buffer_ind = []
        if self._steps == -1:
            for i in range(
                0, len(history.ego_states), round(1 / self._subsample_ratio)
            ):
                buffer_ind.append(i)
                self._steps += 1

            ego_state = history.ego_states[0]
            self._scene.x_min = ego_state.center.x - self._scene_radius
            self._scene.y_min = ego_state.center.y - self._scene_radius
            self._scene.x_max = ego_state.center.x + self._scene_radius
            self._scene.y_max = ego_state.center.y + self._scene_radius

            goal_ego_state = self._expert_goal_state
            self.criterion.set_goal_pos(
                [
                    goal_ego_state.x - self._scene.x_min - self._scene_radius,
                    goal_ego_state.y - self._scene.y_min - self._scene_radius,
                ]
            )

        else:
            buffer_ind.append(-1)
            self._steps += 1

        self._tau = (
            (self._steps - self._last_prediction_time)
            if self._last_prediction_time > 0
            else -1
        )

        ### STEP 1: Add current ego, vehicles, and pedestrians to scene
        for i in buffer_ind:
            ego_state = history.ego_states[i]
            obs = history.observations[i]
            add_observations_to_scene(self._scene, ego_state, obs)

        # ### plot/cost functions
        if self.evaluation and self._tau == self._ph and self._steps > 8:
            print("Evaluating...")
            predictions = self._predictions[self._last_prediction_time]
            scene_name = self._scenario.scenario_name
            new_plot(
                scene_name,
                self._scene,
                predictions,
                self._last_prediction_time,
                self._ph,
            )

        if not self.following_expert and self._tau > 0 and self._steps > 8:
            ### STEP 2: high frequency anomaly detection
            self._achieved_costs[self._steps] = self.criterion.compute_noplan_cost(
                self._scene.robot, np.array([self._steps, self._steps]), None
            )

            ad_rank = anomaly_detection_all(
                self._achieved_costs[self._steps],
                self._predicted_costs[self._last_prediction_time],
                tau=self._tau - 1,
            )

        if (
            self._tau == self._ph
            and (
                self.following_expert or (not self.following_expert and self._steps > 8)
            )
            and self.exp != 4
        ):
            ### STEP 2: low frequency anomaly detection
            prediction_timesteps = np.array(
                [self._last_prediction_time + 1, self._last_prediction_time + self._ph]
            )
            timestep = np.array([self._last_prediction_time])
            with torch.no_grad():
                predictions = self._stg.predict(
                    self._scene,
                    timestep,
                    self._ph,
                    num_samples=self.num_predict_samples,
                )[self._last_prediction_time]
                dists, _ = self._stg.predict(
                    self._scene,
                    timestep,
                    self._ph,
                    num_samples=1,
                    output_dists=True,
                    gmm_mode=True,
                )

            ads = aggregate_anomaly_detection(
                self.criterion,
                self._scene.robot,
                self._ph,
                prediction_timesteps,
                predictions,
                dists[self._last_prediction_time],
            )
            self.advalues.append(ads)
            print(*ads)

        if self.exp == 4 and self._steps - 5 - self._ph == 0:  # only happens once...
            start_of_plan = self._steps - self._ph
            prediction_timesteps = np.array(
                [start_of_plan + 1, start_of_plan + self._ph]
            )
            timestep = np.array([start_of_plan])
            with torch.no_grad():
                predictions = self._stg.predict(
                    self._scene,
                    timestep,
                    self._ph,
                    num_samples=self.num_predict_samples,
                )[start_of_plan]
                dists, _ = self._stg.predict(
                    self._scene,
                    timestep,
                    self._ph,
                    num_samples=1,
                    output_dists=True,
                    gmm_mode=True,
                )
            ads = aggregate_adaptive_planning(
                self.criterion,
                self._scene.robot,
                self._ph,
                prediction_timesteps,
                predictions,
                dists[start_of_plan],
            )
            self.advalues = (ads, len(predictions))
            print(ads)

        ### STEP 3: low frequency step
        if self._tau == self._ph or self._last_prediction_time < 0:

            self._last_prediction_time = self._steps
            self._tau = 0

            if not self.following_expert:
                timestep = np.array([self._steps])
                prediction_timesteps = np.array(
                    [
                        self._last_prediction_time + 1,
                        self._last_prediction_time + self._ph,
                    ]
                )

                ### 3a. Use trajectron to make predictions for other agents
                predictions = self._stg.predict(
                    self._scene,
                    timestep,
                    self._ph,
                    num_samples=self.num_predict_samples,
                )[self._steps]
                self._predictions[self._steps] = predictions

                ### 3b: Make a plan based on predictions
                mp = self.mp_generator.generate_motion_primitives(
                    self._scene.robot, timestep
                )
                ego_nodes = convert_primitives_to_nodes(self._scene.robot, mp)

                best_mp_index = self.criterion.argmin_planning_cost(
                    ego_nodes,
                    prediction_timesteps,
                    self._map_api,
                    predictions,
                    fast=self._fast,
                    verbose=self._verbose,
                    expert_traj=self._expert_traj,
                )
                self._plan = mp[best_mp_index]

                ### 3c: Compute predicted costs given plan
                self._predicted_costs[self._steps] = self.criterion.compute_noplan_cost(
                    ego_nodes[best_mp_index], prediction_timesteps, predictions
                )

        if not self.following_expert:
            ### STEP 4: Execute plan
            plan = self._plan[self._tau : self._tau + 1]
            padding = np.zeros(
                (round(self._future_horizon / self._dt) - plan.shape[0], 3)
            )
            plan = np.concatenate((plan, padding), axis=0)
        else:
            ### STEP 4: Execute empty plan
            plan = np.zeros((round(self._future_horizon / self._dt), 3))

        # Convert relative poses to absolute states and wrap in a trajectory object.
        anchor_ego_state = history.ego_states[-1]
        plan_states = transform_predictions_to_states(
            plan, anchor_ego_state, self._future_horizon, self._dt
        )
        trajectory = InterpolatedTrajectory(plan_states)

        return trajectory
