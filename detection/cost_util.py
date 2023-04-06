from copyreg import constructor
import numpy as np
from detection.query import Query
from detection.query_utils import query_node, query_prediction
from detection.cost_functions import (
    distance_to_goal_cost,
    comfort_cost,
    velocity_cost,
    max_distance_to_agent_cost,
    min_likelihood,
    max_time_to_collision_cost,
    drivable_area_violation,
    lane_violation_cost,
    speed_limit_violation_cost,
    distance_to_agent_cost,
    likelihood,
    total_distance_to_agent_cost,
    total_time_to_collision_cost,
    min_time_gap_to_agent,
    reverse_cost,
    distance_to_goal_scaled_velociy_cost,
    expert_lane_violation_cost,
    min_time_to_collision,
    min_reachability_value,
)
import sys
from multiprocessing import Process, Array

sys.path.append("../trajectron")
from environment.node import Node
from nuplan.planning.utils.multithreading.spliter import chunk_list
import jax.numpy as jnp


class CostUtil:
    def __init__(self, scene):
        self.scene = scene
        self.goal_pos = None

        folder = "../detection/target_values/"
        file1 = "hj_reachability_values_veh.npy"
        data1 = np.load(folder + file1, allow_pickle=True).item()
        grid1 = data1.get("grid")
        target_values1 = data1.get("target_values")
        self.value_fn_veh = lambda state: grid1.interpolate(target_values1, state)

        file2 = "hj_reachability_values_ped.npy"
        data2 = np.load(folder + file2, allow_pickle=True).item()
        grid2 = data2.get("grid")
        target_values2 = data2.get("target_values")
        self.value_fn_ped = lambda state: grid2.interpolate(target_values2, state)

    def set_goal_pos(self, goal_pos):
        self.goal_pos = goal_pos

    def compute_noplan_cost(
        self, ego_node, timesteps, prediction_dict=None, split_agents=False
    ):
        queries = [
            Query.position,
            Query.velocity,
            Query.acceleration,
            Query.heading,
            Query.heading_rate,
            Query.rotated_velocity,
        ]
        ego_states = query_node(ego_node, queries, timesteps)

        if split_agents:
            return max_distance_to_agent_cost(
                ego_node,
                self.scene,
                timesteps,
                predictions=prediction_dict,
                ego_cache=ego_states,
                split_agents=split_agents,
            )

        cost = 0
        cost += max_distance_to_agent_cost(
            ego_node,
            self.scene,
            timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
            split_agents=False,
        )
        cost += max_time_to_collision_cost(
            ego_node,
            self.scene,
            timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
            split_agents=False,
        )

        return cost

    # momentum shaped distance
    def compute_noplan_distance_to_agent_cost(
        self, ego_node, timesteps, prediction_dict=None
    ):
        queries = [
            Query.position,
            Query.velocity,
            Query.acceleration,
            Query.heading,
            Query.heading_rate,
            Query.rotated_velocity,
        ]
        ego_states = query_node(ego_node, queries, timesteps)
        cost = max_distance_to_agent_cost(
            ego_node,
            self.scene,
            timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
            split_agents=False,
        )

        return cost

    def compute_noplan_time_to_collision_cost(
        self, ego_node, timesteps, prediction_dict=None
    ):
        queries = [
            Query.position,
            Query.velocity,
            Query.acceleration,
            Query.heading,
            Query.heading_rate,
            Query.rotated_velocity,
        ]
        ego_states = query_node(ego_node, queries, timesteps)
        cost = max_time_to_collision_cost(
            ego_node,
            self.scene,
            timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
            split_agents=False,
        )

        return cost

    def compute_time_to_collision(self, ego_node, timesteps, prediction_dict=None):
        queries = [
            Query.position,
            Query.velocity,
            Query.acceleration,
            Query.heading,
            Query.heading_rate,
            Query.rotated_velocity,
        ]
        ego_states = query_node(ego_node, queries, timesteps)
        cost = min_time_to_collision(
            ego_node,
            self.scene,
            timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
            split_agents=False,
        )

        return cost

    def compute_likelihood_cost(
        self, ego_node, timesteps, prediction_gmm, split_agents=False
    ):
        queries = [Query.position]
        ego_states = query_node(ego_node, queries, timesteps)
        cost = min_likelihood(
            ego_node,
            self.scene,
            timesteps,
            predictions=prediction_gmm,
            ego_cache=ego_states,
            split_agents=split_agents,
        )

        return cost

    def compute_target_value(self, ego_node, timesteps, prediction_dict=None):
        queries = [Query.position, Query.velocity, Query.heading]
        ego_states = query_node(ego_node, queries, timesteps)
        target_values_veh = self.value_fn_veh
        target_values_ped = self.value_fn_ped
        cost = min_reachability_value(
            ego_node,
            self.scene,
            timesteps,
            predictions=prediction_dict,
            ego_cache=ego_states,
            target_values_veh=target_values_veh,
            target_values_ped=target_values_ped,
        )

        return cost

    def argmin_planning_cost(
        self,
        ego_nodes,
        timesteps,
        map_api,
        prediction_dict,
        fast=False,
        verbose=False,
        expert_traj=None,
    ):
        import time

        scene = self.scene
        goal_pos = self.goal_pos

        s = time.time()

        inds = [i for i in range(len(ego_nodes))]
        results = Array("d", range(len(ego_nodes)))

        ind_chunks = chunk_list(inds)
        ego_node_chunks = chunk_list(ego_nodes)

        proccesses = []
        for ego_node_chunk, ind_chunk in zip(ego_node_chunks, ind_chunks):
            p = Process(
                target=_planning_cost_sub,
                args=(
                    ego_node_chunk,
                    ind_chunk,
                    timesteps,
                    goal_pos,
                    scene,
                    map_api,
                    prediction_dict,
                    results,
                    fast,
                    expert_traj,
                ),
            )
            p.start()
            proccesses.append(p)
        for p in proccesses:
            p.join()

        min_index = np.argmin(results[:])

        return min_index

    def test(self, ego_node, timesteps, map_api, expert_traj):
        expert_lane_violation_cost(ego_node, self.scene, expert_traj, timesteps)
        exit()


import time


def _planning_cost_sub(
    ego_nodes,
    inds,
    timesteps,
    goal_pos,
    scene,
    map_api,
    prediction_dict,
    results,
    fast,
    expert_traj,
):
    queries = [
        Query.position,
        Query.velocity,
        Query.acceleration,
        Query.jerk,
        Query.heading,
        Query.heading_rate,
        Query.heading_acceleration,
        Query.lon_acceleration,
        Query.lat_acceleration,
        Query.lon_jerk,
        Query.jerk_norm,
        Query.rotated_velocity,
        Query.rotated_acceleration,
    ]

    for ego_node, ind in zip(ego_nodes, inds):
        ego_states = query_node(ego_node, queries, timesteps)
        costs = []
        costs.append(
            np.mean(
                max_time_to_collision_cost(
                    ego_node,
                    scene,
                    timesteps,
                    predictions=prediction_dict,
                    ego_cache=ego_states,
                ),
                axis=0,
            )
        )
        costs.append(
            np.mean(
                max_distance_to_agent_cost(
                    ego_node,
                    scene,
                    timesteps,
                    predictions=prediction_dict,
                    ego_cache=ego_states,
                ),
                axis=0,
            )
        )
        costs.append(
            distance_to_goal_cost(ego_node, goal_pos, timesteps, cache=ego_states)
        )
        costs.append(
            distance_to_goal_scaled_velociy_cost(
                ego_node, goal_pos, timesteps, cache=ego_states
            )
        )
        costs.append(comfort_cost(ego_node, timesteps, cache=ego_states))
        costs.append(reverse_cost(ego_node, timesteps, cache=ego_states))

        if not fast:
            # These are slow
            costs.append(
                lane_violation_cost(
                    ego_node, scene, map_api, timesteps, cache=ego_states
                )
            )
        else:
            costs.append(
                expert_lane_violation_cost(
                    ego_node, scene, expert_traj, timesteps, cache=ego_states
                )
            )
        weights = [1, 1, 1, 1, 1, 10, 1]

        total_cost = 0
        for cost, weight in zip(costs, weights):
            total_cost += weight * sum(cost)
        results[ind] = total_cost

    return 1
