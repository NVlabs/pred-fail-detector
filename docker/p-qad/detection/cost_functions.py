import sys

sys.path.append("../")
from detection.query import Query
from detection.query_utils import (
    query_node,
    query_node_and_node,
    query_prediction,
    query_node_and_prediction,
)
import numpy as np
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from trajectron.environment.node import Node
from trajectron.environment.scene import Scene
from trajectron.model.components.gmm2d import GMM2D
from typing import Optional
import shapely.geometry as geom
import jax.numpy as jnp


# Cost functions
def distance_to_goal_cost(node: Node, goal_pos, timesteps, cache=None):
    return distance_to_goal(node, timesteps, goal_pos, cache) / np.linalg.norm(
        goal_pos, axis=-1
    )


def velocity_cost(node: Node, timesteps: np.ndarray, cache=None):
    max_vel = 30
    query = Query.velocity_norm
    states = query_node(node, [query], timesteps, cache)
    vel = states[str(query)]
    return vel**2 / max_vel**2


def comfort_cost(node: Node, timesteps: np.ndarray, cache=None):
    comfort_query = [
        Query.lon_acceleration,
        Query.lat_acceleration,
        Query.lon_jerk,
        Query.jerk_norm,
        Query.heading_rate,
        Query.heading_acceleration,
    ]
    states = query_node(node, comfort_query, timesteps, cache)
    lon_acceleration = states["lon_acceleration"]
    lat_acceleration = states["lat_acceleration"]
    lon_jerk = states["lon_jerk"]
    jerk_norm = states["jerk_norm"]
    heading_rate = states["heading_rate"]
    heading_acceleration = states["heading_acceleration"]

    costs = []
    steps = timesteps[int(len(timesteps) > 1)] - timesteps[0] + 1
    for t in range(steps):
        costs.append(
            _comfort_cost(
                lon_acc=lon_acceleration[t],
                lat_acc=lat_acceleration[t],
                lon_jerk=lon_jerk[t],
                jerk=jerk_norm[t],
                heading_rate=heading_rate[t],
                heading_acc=heading_acceleration[t],
            )
        )
    return costs


def distance_to_agent_cost(
    ego_node: Node,
    agent_node: Node,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_cache=None,
    cache=None,
    **kwargs
):
    vehicle_tuning = (0.5, 0.5)
    pedestrian_tuning = (1, 1)
    added_noise = 0.01

    if agent_node.type.name == "VEHICLE":
        vi_epsilon = vehicle_tuning[0]  # lower means we care more about it
        vo_epsilon = vehicle_tuning[1]
    else:
        vi_epsilon = pedestrian_tuning[0]
        vo_epsilon = pedestrian_tuning[1]

    queries = [Query.rotated_relative_position, Query.rotated_relative_velocity]
    if predictions is None:
        states = query_node_and_node(
            ego_node, agent_node, queries, timesteps, ego_cache, agent_cache, cache
        )
        rotated_relative_position = np.expand_dims(
            states["rotated_relative_position"], axis=0
        )
        rotated_relative_velocity = np.expand_dims(
            states["rotated_relative_velocity"], axis=0
        )
    else:
        states = query_node_and_prediction(
            ego_node,
            agent_node,
            queries,
            timesteps,
            predictions,
            ego_cache,
            agent_cache,
            cache,
        )
        rotated_relative_position = states["rotated_relative_position"]
        rotated_relative_velocity = states["rotated_relative_velocity"]

    returns = _distance_to_agent_cost(
        rotated_relative_position, rotated_relative_velocity, vi_epsilon, vo_epsilon
    )
    if predictions is not None:
        noise = np.random.normal(0, added_noise, size=returns.shape)
        returns += noise

    return returns


def likelihood(
    ego_node: Node,
    agent_node: Node,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_cache=None,
    cache=None,
    **kwargs
):
    prediction_gmm = predictions
    noise_floor = 0
    VV_significance_distance = 20
    VP_significance_distance = 10

    queries = [
        Query.mean_distance_error,
        Query.position_likelihood,
        Query.true_position,
    ]
    agent_cache = query_prediction(
        agent_node, queries, timesteps, prediction_gmm, agent_cache
    )
    dist_error = agent_cache["mean_distance_error"]
    likelihood = agent_cache["position_likelihood"]
    likelihood[dist_error < noise_floor] = np.inf

    query = Query.true_relative_position
    cache = query_node_and_prediction(
        ego_node,
        agent_node,
        [query],
        timesteps,
        prediction_gmm,
        ego_cache,
        agent_cache,
        cache,
    )
    distance = np.linalg.norm(cache["true_relative_position"], axis=-1)

    is_close_enough = np.logical_or(
        np.logical_and(
            agent_node.type.name == "VEHICLE", distance < VV_significance_distance
        ),
        np.logical_and(
            agent_node.type.name == "PEDESTRIAN", distance < VP_significance_distance
        ),
    )

    likelihood[~is_close_enough] = np.inf

    return np.array([likelihood])


def drivable_area_violation(
    node: Node, scene: Scene, map_api: AbstractMap, timesteps, cache=None
):
    position_shift = [
        scene.x_min + (scene.x_max - scene.x_min) / 2,
        scene.y_min + (scene.y_max - scene.y_min) / 2,
    ]
    query = Query.position
    cache = query_node(node, [query], timesteps, states=cache)
    positions = cache[str(query)] + position_shift
    violation = []
    for position in positions:
        point = Point2D(*position)
        _, distance_to_drivable_area = map_api.get_distance_to_nearest_map_object(
            point, SemanticMapLayer.DRIVABLE_AREA
        )
        violation.append(float(distance_to_drivable_area))

    return violation


def lane_violation_cost(
    node: Node, scene: Scene, map_api: AbstractMap, timesteps: np.ndarray, cache=None
):
    position_shift = [
        scene.x_min + (scene.x_max - scene.x_min) / 2,
        scene.y_min + (scene.y_max - scene.y_min) / 2,
    ]
    query = [Query.position, Query.heading]
    cache = query_node(node, query, timesteps, states=cache)
    positions = cache["position"] + position_shift
    headings = cache["heading"]

    cost = []
    for position, heading in zip(positions, headings):
        point = Point2D(*position)
        obj = map_api.get_one_map_object(point, SemanticMapLayer.LANE)

        if map_api.is_in_layer(point, SemanticMapLayer.LANE):
            _, dist_to_baseline = map_api.get_distance_to_nearest_map_object(
                point, SemanticMapLayer.BASELINE_PATHS
            )
            obj = map_api.get_one_map_object(point, SemanticMapLayer.LANE)
        elif map_api.is_in_layer(point, SemanticMapLayer.INTERSECTION):
            name, dist_to_baseline = map_api.get_distance_to_nearest_map_object(
                point, SemanticMapLayer.LANE_CONNECTOR
            )
            obj = map_api.get_map_object(name, SemanticMapLayer.LANE_CONNECTOR)
        else:
            _, dist_to_baseline1 = map_api.get_distance_to_nearest_map_object(
                point, SemanticMapLayer.BASELINE_PATHS
            )
            name, dist_to_baseline2 = map_api.get_distance_to_nearest_map_object(
                point, SemanticMapLayer.LANE_CONNECTOR
            )
            dist_to_baseline = min(dist_to_baseline1, dist_to_baseline2) * 10
            if dist_to_baseline1 < dist_to_baseline2:
                dist_to_baseline = dist_to_baseline1
                name, _ = map_api.get_distance_to_nearest_map_object(
                    point, SemanticMapLayer.LANE
                )
                obj = map_api.get_map_object(name, SemanticMapLayer.LANE)
            else:
                dist_to_baseline = dist_to_baseline2
                obj = map_api.get_map_object(name, SemanticMapLayer.LANE_CONNECTOR)

        dist_to_baseline_cost = dist_to_baseline**4 / 8

        blp = obj.baseline_path()
        goal_heading = blp.get_nearest_pose_from_position(point).heading
        heading_cost = (goal_heading - heading) % (2 * np.pi)
        heading_cost -= 2 * np.pi if heading_cost > np.pi else 0
        heading_cost = heading_cost**2 / 2

        cost.append(dist_to_baseline_cost + heading_cost)

    return cost


def expert_lane_violation_cost(
    node: Node, scene: Scene, expert_traj: np.ndarray, timesteps: np.ndarray, cache=None
):
    position_shift = [
        scene.x_min + (scene.x_max - scene.x_min) / 2,
        scene.y_min + (scene.y_max - scene.y_min) / 2,
    ]
    query = [Query.position, Query.heading]
    cache = query_node(node, query, timesteps, states=cache)
    positions = cache["position"] + position_shift
    headings = cache["heading"]

    cost = []
    for position, heading in zip(positions, headings):
        diff = np.linalg.norm(expert_traj[:, :2] - position, axis=-1)
        closest_index = np.argmin(diff)
        next_index = closest_index + 1 if closest_index < 39 else closest_index - 1

        closest_point = expert_traj[closest_index]
        next_point = expert_traj[next_index]

        x2 = next_point[0]
        y2 = next_point[1]
        x1 = closest_point[0]
        y1 = closest_point[1]
        dist_to_baseline = abs(
            (x2 - x1) * (y1 - position[1]) - (y2 - y1) * (x1 - position[0])
        ) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        dist_to_baseline_cost = dist_to_baseline**4 / 4

        heading_cost = (closest_point[2] - heading) % (2 * np.pi)
        heading_cost -= 2 * np.pi if heading_cost > np.pi else 0
        heading_cost = heading_cost**2 / 2

        cost.append(dist_to_baseline_cost + heading_cost)

    return cost


def reverse_cost(node: Node, timesteps: np.ndarray, cache=None):
    query = Query.lon_velocity
    v = query_node(node, [query], timesteps, states=cache)[str(query)]
    v[v >= 0] = 0
    v[v < 0] = 1
    return v


def speed_limit_violation_cost(
    node: Node, scene: Scene, map_api: AbstractMap, timesteps: np.ndarray, cache=None
):
    position_shift = [
        scene.x_min + (scene.x_max - scene.x_min) / 2,
        scene.y_min + (scene.y_max - scene.y_min) / 2,
    ]
    queries = [Query.position, Query.velocity_norm]
    cache = query_node(node, queries, timesteps, states=cache)
    positions = cache["position"] + position_shift
    speeds = cache["velocity_norm"]
    violation = []
    layer = None
    for position, speed in zip(positions, speeds):
        point = Point2D(*position)
        if map_api.is_in_layer(point, SemanticMapLayer.LANE):
            layer = SemanticMapLayer.LANE
        elif map_api.is_in_layer(point, SemanticMapLayer.INTERSECTION):
            layer = SemanticMapLayer.LANE_CONNECTOR

        if layer is not None:
            segments = map_api.get_all_map_objects(point, layer)
            if len(segments) > 0:
                segment = segments[0]
                speed_limit = segment.speed_limit_mps
                scaled_v = (speed - speed_limit) / (speed_limit / 2)
                violation.append(max(0, scaled_v))
            else:
                violation.append(0.0)
        else:
            violation.append(0.0)

    return violation


def distance_to_goal_scaled_velociy_cost(
    node: Node, goal_pos, timesteps: np.ndarray, cache=None
):

    total_scene_length = 20  # including history

    goal_vel = (
        np.linalg.norm(goal_pos, axis=-1) / total_scene_length
    ) * 0.8  # go a bit slower

    additional_allowable_velocity = max(goal_vel * 0.1, 1)
    max_vel = np.maximum(30 - goal_vel, 10)
    queries = [Query.velocity_norm, Query.position]
    states = query_node(node, queries, timesteps, cache)
    vel = states["velocity_norm"]

    vel_cost = np.maximum(np.abs(vel - goal_vel) - additional_allowable_velocity, 0)

    return vel_cost**2 / max_vel**2


# Metrics
def distance_to_goal(node: Node, timesteps: np.ndarray, goal_pos, cache=None):
    query = Query.position
    states = query_node(node, [query], timesteps, cache)
    state = states[str(query)]
    return np.linalg.norm(state - goal_pos, axis=-1)


def euclidean_distance(
    ego_node: Node,
    agent_node: Node,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_cache=None,
    cache=None,
):
    query = Query.relative_position
    if predictions is None:
        states = query_node_and_node(
            ego_node, agent_node, [query], timesteps, ego_cache, agent_cache, cache
        )
    else:
        states = query_node_and_prediction(
            ego_node,
            agent_node,
            [query],
            timesteps,
            predictions,
            ego_cache,
            agent_cache,
            cache,
        )
    return np.linalg.norm(states[str(query)], axis=-1)


def time_to_collision(
    ego_node: Node,
    agent_node: Node,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_cache=None,
    cache=None,
    **kwargs
):
    max_value = kwargs["max_value"]
    if max_value is not None and round(max_value) == 50:
        max_ttc = 50
        added_noise = 0
    else:
        max_ttc = 3.0
        added_noise = max_ttc / 10

    if agent_node.type.name == "VEHICLE":
        r = 2.0
    else:
        r = 1.2
    ttc = np.array([max_ttc for _ in range(timesteps[1] - timesteps[0] + 1)])

    first_dim = timesteps[1] - timesteps[0] + 1
    if predictions is None:
        ttc = np.zeros((1, first_dim)) + max_ttc
    else:
        second_dim = predictions[ego_node].shape[1]
        ttc = np.zeros((second_dim, first_dim)) + max_ttc

    queries = [Query.relative_position, Query.relative_velocity]
    if predictions is None:
        cache = query_node_and_node(
            ego_node, agent_node, queries, timesteps, ego_cache, agent_cache, cache
        )
        rp = np.expand_dims(cache["relative_position"], axis=0)
        rv = np.expand_dims(cache["relative_velocity"], axis=0)
    else:
        cache = query_node_and_prediction(
            ego_node,
            agent_node,
            queries,
            timesteps,
            predictions,
            ego_cache,
            agent_cache,
            cache,
        )
        rp = cache["relative_position"]
        rv = cache["relative_velocity"]

    rr = 2 * r
    rm = np.divide(rv[:, :, 1], rv[:, :, 0])
    rb = rp[:, :, 1] - rm * rp[:, :, 0]
    xclose = -rb * rm / (rm**2 + 1)
    yclose = rb / (rm**2 + 1)
    pclose = np.stack((xclose, yclose), axis=-1)
    dist_close = np.linalg.norm(pclose, axis=-1)
    t_to_closest_point = ((pclose - rp) / rv)[:, :, 0]
    t_on_line = np.sqrt(rr**2 - dist_close**2) / np.linalg.norm(rv, axis=-1)

    t_to_collision = t_to_closest_point - t_on_line  # nan if there is no collision

    collision_occurs = np.logical_and(t_to_closest_point > 0, dist_close < rr)
    ttc[collision_occurs] = np.minimum(
        ttc[collision_occurs], t_to_collision[collision_occurs]
    )
    if predictions is not None:
        noise = np.random.normal(0, added_noise, size=ttc.shape)
        ttc += noise
    return ttc


def time_gap_to_agent(
    ego_node: Node,
    agent_node: Node,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_cache=None,
    cache=None,
    **kwargs
):
    added_noise = 0.3
    ego_query = Query.velocity_norm
    ego_cache = query_node(ego_node, [ego_query], timesteps, states=ego_cache)
    speed = ego_cache["velocity_norm"]

    query = Query.relative_position
    if predictions is None:
        cache = query_node_and_node(
            ego_node, agent_node, [query], timesteps, ego_cache, agent_cache, cache
        )
        relative_position = cache["relative_position"]
        relative_position = np.expand_dims(relative_position, axis=0)
    else:
        cache = query_node_and_prediction(
            ego_node,
            agent_node,
            [query],
            timesteps,
            predictions,
            ego_cache,
            agent_cache,
            cache,
        )
        relative_position = cache["relative_position"]

    tg = np.linalg.norm(relative_position, axis=-1) / speed
    if predictions is not None:
        noise = np.random.normal(0, added_noise, size=tg.shape)
        tg += noise
    return tg


def reachability_value(
    ego_node: Node,
    agent_node: Node,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_cache=None,
    cache=None,
    **kwargs
):
    if agent_node.type.name == "VEHICLE":
        target_values = kwargs["target_values_veh"]
    else:
        target_values = kwargs["target_values_ped"]

    queries = [Query.lon_velocity, Query.heading]
    ego_cache = query_node(ego_node, queries, timesteps, states=ego_cache)
    ego_vel = ego_cache["lon_velocity"]

    if agent_node.type.name == "VEHICLE":
        agent_cache = query_node(agent_node, queries, timesteps, states=agent_cache)
        agent_vel = agent_cache["lon_velocity"]
        r_heading = agent_cache["heading"] - ego_cache["heading"]
    else:
        agent_vel = [0] * len(ego_vel)  # agent_cache['velocity_norm']
        r_heading = ego_cache["heading"]

    queries = [Query.relative_position]
    cache = query_node_and_node(
        ego_node, agent_node, queries, timesteps, ego_cache, agent_cache, cache
    )
    r_state = cache["relative_position"]

    r_heading = [(h - 2 * np.pi if h > np.pi else h) for h in r_heading]
    relative_states = np.array(
        [r_state[:, 0], r_state[:, 1], r_heading, ego_vel, agent_vel]
    )
    value = []
    for i in range(len(relative_states[0, :])):
        relative_state = relative_states[:, i]
        if (
            abs(relative_state[0]) > 10
            or abs(relative_state[1]) > 10
            or relative_state[3] > 18
            or relative_state[4] > 18
            or np.isnan(np.sum(relative_state))
        ):
            value.append(np.inf)
            continue
        elif np.isnan(np.sum(relative_state)):
            value.append(np.inf)
            continue
        if relative_state[3] < 0:
            relative_state[3] = 0
        if relative_state[4] < 0:
            relative_state[4] = 0

        value.append(target_values(jnp.array(relative_state)))

    return np.array([value])


### Extrema over scene:
def max_distance_to_agent_cost(
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
):
    min_distance_cost = 0
    max_dc = _extrema_over_scene(
        distance_to_agent_cost,
        np.maximum,
        min_distance_cost,
        ego_node,
        scene,
        timesteps,
        predictions,
        ego_cache,
        agent_caches,
        caches,
        split_agents,
    )
    return max_dc


def total_distance_to_agent_cost(
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
):
    initial_distance_cost = 0
    max_dc = _extrema_over_scene(
        distance_to_agent_cost,
        np.add,
        initial_distance_cost,
        ego_node,
        scene,
        timesteps,
        predictions,
        ego_cache,
        agent_caches,
        caches,
        split_agents,
    )
    return max_dc


def max_time_to_collision_cost(
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
):
    max_ttc = 100.0
    min_ttc = _extrema_over_scene(
        time_to_collision,
        np.minimum,
        max_ttc,
        ego_node,
        scene,
        timesteps,
        predictions,
        ego_cache,
        agent_caches,
        caches,
        split_agents,
    )

    if split_agents:
        for node in min_ttc.keys():
            min_ttc[node] = 1 - min_ttc[node] / max_ttc
    else:
        min_ttc = 1 - min_ttc / max_ttc

    return min_ttc


def min_time_to_collision(
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
):
    max_ttc = 50.0
    min_ttc = _extrema_over_scene(
        time_to_collision,
        np.minimum,
        max_ttc,
        ego_node,
        scene,
        timesteps,
        predictions,
        ego_cache,
        agent_caches,
        caches,
        split_agents,
    )
    return min_ttc


def min_reachability_value(
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
    target_values_veh=None,
    target_values_ped=None,
):
    max_value = np.inf
    min_value = _extrema_over_scene(
        reachability_value,
        np.minimum,
        max_value,
        ego_node,
        scene,
        timesteps,
        predictions,
        ego_cache,
        agent_caches,
        caches,
        split_agents,
        target_values_veh=target_values_veh,
        target_values_ped=target_values_ped,
    )
    return min_value


def total_time_to_collision_cost(
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
):
    max_ttc = 3.0
    min_ttc = _extrema_over_scene(
        time_to_collision,
        np.minimum,
        max_ttc,
        ego_node,
        scene,
        timesteps,
        predictions,
        ego_cache,
        agent_caches,
        caches,
        split_agents=True,
    )

    if split_agents:
        for node in min_ttc.keys():
            min_ttc[node] = 1 - min_ttc[node] / max_ttc
        return min_ttc
    else:
        total_ttc_cost = 0
        for node in min_ttc.keys():
            total_ttc_cost += 1 - min_ttc[node] / max_ttc
        return total_ttc_cost


def min_time_gap_to_agent(
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
):
    max_time_gap = 3
    min_time_gap = _extrema_over_scene(
        time_gap_to_agent,
        np.minimum,
        max_time_gap,
        ego_node,
        scene,
        timesteps,
        predictions,
        ego_cache,
        agent_caches,
        caches,
        split_agents,
    )

    return min_time_gap


def min_likelihood(
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
):
    max_llhd = np.inf
    min_llhd = _extrema_over_scene(
        likelihood,
        np.minimum,
        max_llhd,
        ego_node,
        scene,
        timesteps,
        predictions,
        ego_cache,
        agent_caches,
        caches,
        split_agents,
    )
    return min_llhd


def _extrema_over_scene(
    function,
    operation,
    initial_extrema,
    ego_node: Node,
    scene: Scene,
    timesteps: np.ndarray,
    predictions=None,
    ego_cache=None,
    agent_caches=None,
    caches=None,
    split_agents=False,
    **kwargs
):
    if split_agents:
        extrema_result = {}
    else:
        first_dim = timesteps[1] - timesteps[0] + 1
        if predictions is None or not hasattr(predictions[ego_node], "shape"):
            extrema_result = np.zeros((1, first_dim)) + initial_extrema
        else:
            second_dim = predictions[ego_node].shape[1]
            extrema_result = np.zeros((second_dim, first_dim)) + initial_extrema

    if agent_caches is None:
        agent_caches = {}
    if caches is None:
        caches = {}

    pred_nodes = None
    if predictions is not None:
        pred_nodes = predictions.keys()

    for agent_node in scene.nodes:
        if predictions is not None and agent_node not in pred_nodes:
            continue
        if agent_node.id == "ego":
            continue
        if agent_node not in agent_caches:
            agent_caches[agent_node] = None
        if (ego_node, agent_node) not in caches:
            caches[(ego_node, agent_node)] = None

        result = function(
            ego_node,
            agent_node,
            timesteps,
            predictions=predictions,
            ego_cache=ego_cache,
            agent_cache=agent_caches[agent_node],
            cache=caches[(ego_node, agent_node)],
            max_value=initial_extrema,
            **kwargs
        )
        valid_ind = ~np.isnan(result)

        if split_agents:
            result[~valid_ind] = initial_extrema
            extrema_result[agent_node] = result
        else:
            extrema_result[valid_ind] = operation(
                extrema_result[valid_ind], result[valid_ind]
            )

    return extrema_result


### Metrics/costs which don't use queries
def _comfort_cost(lon_acc, lat_acc, lon_jerk, jerk, heading_rate, heading_acc):
    min_lon_acc = -4.05
    max_lon_acc = 2.40
    max_lat_acc = 4.89
    max_lon_jerk = 4.13
    max_jerk = 8.37
    max_heading_rate = 0.95
    max_heading_acc = 1.93
    weight_list = 6 * [1 / 6]

    lon_acc_cost = max(min_lon_acc - lon_acc, lon_acc - max_lon_acc, 0) / (
        max_lon_acc - min_lon_acc
    )
    lat_acc_cost = max(abs(lat_acc) - max_lat_acc, 0) / (2 * max_lat_acc)
    lon_jerk_cost = max(abs(lon_jerk) - max_lon_jerk, 0) / (2 * max_lon_jerk)
    jerk_cost = max(abs(jerk) - max_jerk, 0) / (2 * max_jerk)
    heading_rate_cost = max(abs(heading_rate) - max_heading_rate, 0) / (
        2 * max_heading_rate
    )
    heading_acc_cost = max(abs(heading_acc) - max_heading_acc, 0) / (
        2 * max_heading_acc
    )

    cost_list = [
        lon_acc_cost,
        lat_acc_cost,
        lon_jerk_cost,
        jerk_cost,
        heading_rate_cost,
        heading_acc_cost,
    ]
    total_cost = sum(
        [_cost * _weight for _cost, _weight in zip(cost_list, weight_list)]
    )
    return total_cost


def _distance_to_agent_cost(
    rotated_relative_position, rotated_relative_velocity, vi_epsilon, vo_epsilon
):
    exp_rotated_relative_velocity = 1 + np.exp(rotated_relative_velocity)
    rotated_relative_position = rotated_relative_position**2
    expterm = (
        rotated_relative_position[:, :, 0]
        * exp_rotated_relative_velocity[:, :, 0]
        * vi_epsilon
        + rotated_relative_position[:, :, 1]
        * exp_rotated_relative_velocity[:, :, 1]
        * vo_epsilon
    )
    rbf = np.exp(-0.5 * expterm)

    return rbf * 10


## Not implemented
def clearance_from_static_agents(*args, **kwargs):
    raise NotImplementedError


def ego_collisions(*args, **kwargs):
    raise NotImplementedError
