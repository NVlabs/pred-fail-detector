import sys, os, dill, json
from tokenize import String
import numpy as np
from nuplan_utils import generate_token
from typing import Optional
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

sys.path.append("../trajectron")
from model.model_registrar import ModelRegistrar
from model import Trajectron
from environment.environment import Environment
from environment.environment import NodeTypeEnum
from environment.scene import Scene
from environment.node import Node
from environment.data_structures import DoubleHeaderNumpyArray
from environment.node_type import NodeType

NODE_TYPE_LIST = ["VEHICLE", "PEDESTRIAN"]  # ['PEDESTRIAN', 'BICYCLE', 'VEHICLE']
PEDESTRIAN = NodeType(name="PEDESTRIAN", value=2)
VEHICLE = NodeType(name="VEHICLE", value=1)
HEADER = [
    ("position", "x"),
    ("position", "y"),
    ("velocity", "x"),
    ("velocity", "y"),
    ("acceleration", "x"),
    ("acceleration", "y"),
    ("heading", "x"),
    ("heading", "y"),
    ("heading", "°"),
    ("heading", "d°"),
    ("velocity", "norm"),
    ("acceleration", "norm"),
]
STANDARDIZATION = {
    PEDESTRIAN: {
        "position": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity": {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    },
    VEHICLE: {
        "position": {"x": {"mean": 0, "std": 80}, "y": {"mean": 0, "std": 80}},
        "velocity": {
            "x": {"mean": 0, "std": 15},
            "y": {"mean": 0, "std": 15},
            "norm": {"mean": 0, "std": 15},
        },
        "acceleration": {
            "x": {"mean": 0, "std": 4},
            "y": {"mean": 0, "std": 4},
            "norm": {"mean": 0, "std": 4},
        },
        "heading": {
            "x": {"mean": 0, "std": 1},
            "y": {"mean": 0, "std": 1},
            "°": {"mean": 0, "std": 3.141592653589793},
            "d°": {"mean": 0, "std": 1},
        },
    },
}
ATTENTION_RADIUS = {
    (PEDESTRIAN, PEDESTRIAN): 10.0,
    (PEDESTRIAN, VEHICLE): 20.0,
    (VEHICLE, PEDESTRIAN): 20.0,
    (VEHICLE, VEHICLE): 30.0,
}
DT = 0.5
LABELMAP = {
    0: "generic_object",
    1: VEHICLE,
    2: PEDESTRIAN,
    3: "bike",
    4: "traffic_cone",
    5: "barrier",
    6: "czone_sign",
}


def build_trajectron_env():
    timesteps = 0
    placeholder_scene = [Scene(timesteps=timesteps, dt=DT)]
    env = Environment(
        scenes=placeholder_scene,
        node_type_list=NODE_TYPE_LIST,
        standardization=STANDARDIZATION,
        attention_radius=ATTENTION_RADIUS,
        robot_type=VEHICLE,
        dt=DT,
    )

    return env


def _x_radius(scene: Scene):
    return (scene.x_max - scene.x_min) / 2


def _y_radius(scene: Scene):
    return (scene.y_max - scene.y_min) / 2


def position_shift(scene: Scene):
    return [scene.x_min + _x_radius(scene), scene.y_min + _y_radius(scene)]


def load_trajectron_model(eval_env, ts=12):

    model_dir = "../models/int_ee"
    with open(os.path.join(model_dir, "config.json"), "r") as config_json:
        trajectron_hyperparams = json.load(config_json)

    model_registrar = ModelRegistrar(model_dir, "cpu")
    model_registrar.load_models(ts)
    trajectron_hyperparams["map_enc_dropout"] = 0.0
    if "incl_robot_node" not in trajectron_hyperparams:
        trajectron_hyperparams["incl_robot_node"] = False
    stg = Trajectron(model_registrar, trajectron_hyperparams, None, "cpu")
    stg.set_environment(eval_env)
    stg.set_annealing_params()

    return stg


def add_observations_to_scene(
    scene: Scene, ego_state: EgoState, observation: Observation
):
    add_agent_to_scene(scene, ego_state, is_ego=True)

    for box in observation.boxes:
        if LABELMAP[box.label] == VEHICLE or LABELMAP[box.label] == PEDESTRIAN:
            add_agent_to_scene(scene, box, is_ego=False)

    scene.timesteps += 1


def add_agent_to_scene(scene: Scene, agent_data: Box3D, is_ego=False):
    node = find_node(scene, "ego" if is_ego else agent_data.track_token, agent_data)

    if node is None or node == -1:
        if node == -1:
            token = generate_token()
        else:
            token = "ego" if is_ego else agent_data.track_token
        data = DoubleHeaderNumpyArray(np.array([]), HEADER)
        node = Node(
            node_type=LABELMAP[1 if is_ego else agent_data.label],
            node_id=token,
            data=data,
            first_timestep=scene.timesteps,
            is_robot=is_ego,
        )
        scene.nodes.append(node)

    lag = scene.timesteps - (node.first_timestep + len(node.data.data))
    assert lag == 0, "Implement method for dealing with interrupted detection"

    data = (
        convert_ego_state_to_node(scene, agent_data, node)
        if is_ego
        else convert_box_to_node(scene, agent_data, node)
    )

    if len(node.data.data.shape) == 1:  # empty
        node.data.data = np.array([data])
    else:
        node.data.data = np.append(node.data.data, [data], axis=0)

    node._last_timestep = node.first_timestep + node.timesteps - 1

    if is_ego:
        scene.robot = node


def find_node(scene: Scene, token: str, agent_data=None):
    nodes = scene.nodes
    if token is None and agent_data is not None:  # simple tracking for reactive agents
        assert (
            token is not None
        ), "I added tokens to reactive agents, this should not be accessed."
        closest_feasible_node = -1
        min_track_metric = np.inf
        for node in nodes:
            lag = scene.timesteps - (node.first_timestep + len(node.data.data))
            dist = np.linalg.norm(
                agent_data.center[:-1]
                - (node.data.data[-1, :2] + position_shift(scene))
            )
            orientation_diff = abs(
                agent_data.orientation.yaw_pitch_roll[0] - node.data.data[-1, 8]
            )
            if lag == 0 and dist < 10 and orientation_diff < np.pi / 6:
                track_metric = dist / 5 + orientation_diff
                if track_metric < min_track_metric:
                    min_track_metric = track_metric
                    closest_feasible_node = node

        return closest_feasible_node

    for node in nodes:
        if token == node.id:
            return node

    return None


def convert_box_to_node(scene: Scene, box: Box3D, node: Node):
    dt = scene.dt
    x_position = box.center[0] - scene.x_min - _x_radius(scene)
    y_position = box.center[1] - scene.y_min - _y_radius(scene)

    heading = box.orientation.yaw_pitch_roll[0]
    if heading > np.pi:
        heading -= np.pi

    if len(node.data.data) == 0:
        x_velocity = 0
        y_velocity = 0
        d_heading = 0
        norm_velocity = 0
        x_heading = 0
        y_heading = 0
    else:
        x_velocity = (x_position - node.data.data[-1, 0]) / dt
        y_velocity = (y_position - node.data.data[-1, 1]) / dt
        d_heading = (heading - node.data.data[-1, 8]) / dt
        norm_velocity = np.sqrt(x_velocity**2 + y_velocity**2)
        x_heading = x_velocity / norm_velocity
        y_heading = y_velocity / norm_velocity
    if len(node.data.data) <= 1:
        x_acceleration = 0
        y_acceleration = 0
        norm_acceleration = 0
    else:
        x_acceleration = (x_velocity - node.data.data[-1, 2]) / dt
        y_acceleration = (y_velocity - node.data.data[-1, 3]) / dt
        norm_acceleration = np.sqrt(x_acceleration**2 + y_acceleration**2)

    data = np.array(
        [
            x_position,
            y_position,
            x_velocity,
            y_velocity,
            x_acceleration,
            y_acceleration,
            x_heading,
            y_heading,
            heading,
            d_heading,
            norm_velocity,
            norm_acceleration,
        ]
    )

    return data


def convert_ego_state_to_node(
    scene: Scene, ego_state: EgoState, ego_node: Optional[Node] = None
):
    x_position = ego_state.center.x - scene.x_min - _x_radius(scene)
    y_position = ego_state.center.y - scene.y_min - _y_radius(scene)
    heading = ego_state.center.heading
    cc = np.cos(heading)
    ss = np.sin(heading)

    lon_vel = ego_state.dynamic_car_state.center_velocity_2d.x
    lat_vel = ego_state.dynamic_car_state.center_velocity_2d.y
    x_velocity = lon_vel * cc - lat_vel * ss
    y_velocity = lon_vel * ss + lat_vel * cc

    if abs(x_velocity) < 1e-6 and abs(y_velocity) < 1e-6 and ego_node is not None:
        x_velocity = (x_position - ego_node.data.data[-1, 0]) / DT
        y_velocity = (y_position - ego_node.data.data[-1, 1]) / DT
        x_acceleration = (x_velocity - ego_node.data.data[-1, 2]) / DT
        y_acceleration = (y_velocity - ego_node.data.data[-1, 3]) / DT
        d_heading = (heading - ego_node.data.data[-1, 8]) / DT
        norm_velocity = np.sqrt(x_velocity**2 + y_velocity**2)
        norm_acceleration = np.sqrt(x_acceleration**2 + y_acceleration**2)
    else:
        lon_acc = ego_state.dynamic_car_state.center_acceleration_2d.x
        lat_acc = ego_state.dynamic_car_state.center_acceleration_2d.y
        x_acceleration = lon_acc * cc - lat_acc * ss
        y_acceleration = lon_acc * ss + lat_acc * cc
        d_heading = ego_state.dynamic_car_state.angular_velocity
        norm_velocity = ego_state.dynamic_car_state.speed
        norm_acceleration = ego_state.dynamic_car_state.acceleration

    x_heading = x_velocity / norm_velocity
    y_heading = y_velocity / norm_velocity

    data = np.array(
        [
            x_position,
            y_position,
            x_velocity,
            y_velocity,
            x_acceleration,
            y_acceleration,
            x_heading,
            y_heading,
            heading,
            d_heading,
            norm_velocity,
            norm_acceleration,
        ]
    )

    return data


def convert_primitives_to_nodes(node: Node, primitives: np.ndarray):
    num_nodes = primitives.shape[0]
    ph = primitives.shape[1]

    node_data = np.array([node.data.data] * num_nodes)

    for i in range(ph):
        h = node_data[:, -1, 8] + primitives[:, i, 2]
        rx = primitives[:, i, 0] * np.cos(h) - primitives[:, i, 1] * np.sin(h)
        ry = primitives[:, i, 0] * np.sin(h) + primitives[:, i, 1] * np.cos(h)
        x = node_data[:, -1, 0] + rx
        y = node_data[:, -1, 1] + ry
        vx = rx / DT
        vy = ry / DT
        ax = (vx - node_data[:, -1, 2]) / DT
        ay = (vy - node_data[:, -1, 3]) / DT
        hx = x * 0
        hy = x * 0
        dh = primitives[:, i, 2] / DT
        vn = np.sqrt(vx**2 + vy**2)
        an = np.sqrt(ax**2 + ay**2)

        ap = np.expand_dims(
            np.array([x, y, vx, vy, ax, ay, hx, hy, h, dh, vn, an]).T, axis=1
        )
        node_data = np.concatenate((node_data, ap), axis=1)

    nodes = []
    for i in range(num_nodes):
        data = DoubleHeaderNumpyArray(node_data[i], HEADER)
        nodes.append(
            Node(
                node_type=VEHICLE,
                node_id="ego",
                data=data,
                first_timestep=0,
                is_robot=True,
            )
        )

    return nodes


def serialize_ego_trajectory(scenario: AbstractScenario, subsample_ratio=0.1):
    max_data_frequency = 20
    scenario_length = 20
    num_time_steps = round(max_data_frequency * scenario_length * subsample_ratio)
    ego_traj = np.zeros((num_time_steps, 3))
    for i in range(num_time_steps):
        center = scenario.get_ego_state_at_iteration(i).center
        ego_traj[i] = [center.x, center.y, center.heading]
    return ego_traj
