from re import I
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import warnings

warnings.filterwarnings("ignore")
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as pe
from scipy.ndimage import rotate
import seaborn as sns
from utils import prediction_output_to_trajectories


# Global plotting vars
layers = [
    "drivable_area",
    "road_segment",
    "lane",
    "ped_crossing",
    "walkway",
    "stop_line",
    "road_divider",
    "lane_divider",
]

cars = [
    plt.imread("icons/Car TOP_VIEW 375397.png"),
    plt.imread("icons/Car TOP_VIEW F05F78.png"),
    plt.imread("icons/Car TOP_VIEW 80CBE5.png"),
    plt.imread("icons/Car TOP_VIEW ABCB51.png"),
    plt.imread("icons/Car TOP_VIEW C8B0B0.png"),
]

line_colors = ["#375397", "#F05F78", "#80CBE5", "#ABCB51", "#C8B0B0"]

line_alpha = 0.7
line_width = 0.2
edge_width = 2
circle_edge_width = 0.5
node_circle_size = 0.3
zoom_scale = 2.0

circle_ads = False


def plot_predictions(
    ax,
    predictions,
    dt,
    max_hl=10,
    ph=6,
    map=None,
    my_patch=(0, 0, 0, 0),
    tau=0,
    ad_dict=None,
):
    x_min, y_min, x_max, y_max = my_patch
    zoom = zoom_scale * min(1.6 / (x_max - x_min), 1.6 / (y_max - y_min))

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(
        predictions, dt, max_hl, ph, map=map
    )
    assert len(prediction_dict.keys()) <= 1
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.fdata, origin="lower", alpha=0.5)

    node_list = sorted(histories_dict.keys(), key=lambda x: x.id)

    for node in node_list:
        history = histories_dict[node] + np.array([x_min, y_min])
        future = futures_dict[node] + np.array([x_min, y_min])
        predictions = prediction_dict[node] + np.array([x_min, y_min])

        color = "r" if node.id == "ego" else "b"

        sample_num = 0
        hj_nodes = [
            "22652adf8d4740fe8bae510d27aecb70",  # 29
            "4b4a5d32d507421cab3c74024778e621",  # 72
            "4d13616574ad4e369ca4237e6d59a528",  # 111
        ]
        if node.type.name == "VEHICLE":
            if (
                not node.id == "ego" and (ad_dict is None or (ad_dict[node] > 0.99))
            ) or node.id in hj_nodes:
                for t in range(tau, predictions.shape[2]):
                    xavg = np.mean(predictions[0, :, t, 0])
                    yavg = np.mean(predictions[0, :, t, 1])
                    sns.kdeplot(
                        predictions[0, :, t, 0],
                        predictions[0, :, t, 1],
                        ax=ax,
                        shade=True,
                        shade_lowest=False,
                        color=color,
                        zorder=600,
                        alpha=0.8,
                    )
                    ax.plot(
                        xavg,
                        yavg,
                        "ko-",
                        zorder=620,
                        markersize=5,
                        linewidth=3,
                        alpha=0.7,
                    )

            ax.plot(
                future[tau:, 0],
                future[tau:, 1],
                "w--o",
                linewidth=4,
                markersize=3,
                zorder=650,
                path_effects=[pe.Stroke(linewidth=5, foreground="k"), pe.Normal()],
            )
        else:
            if ad_dict is None or (ad_dict[node] > 0.99):
                for t in range(tau, predictions.shape[2]):
                    xavg = np.mean(predictions[0, :, t, 0])
                    yavg = np.mean(predictions[0, :, t, 1])
                    sns.kdeplot(
                        predictions[0, :, t, 0],
                        predictions[0, :, t, 1],
                        ax=ax,
                        shade=True,
                        shade_lowest=False,
                        color=color,
                        zorder=600,
                        alpha=0.8,
                    )
                    ax.plot(
                        xavg,
                        yavg,
                        "ko-",
                        zorder=620,
                        markersize=2,
                        linewidth=1,
                        alpha=0.7,
                    )

            ax.plot(
                future[tau:, 0],
                future[tau:, 1],
                "w--",
                zorder=650,
                path_effects=[
                    pe.Stroke(linewidth=edge_width, foreground="k"),
                    pe.Normal(),
                ],
            )


def plot_agents(
    ax, eval_stg, scene, timestep, my_patch=(0, 0, 0, 0), zoom=1, ad_dict=None
):
    x_min, y_min, x_max, y_max = my_patch
    x_span = x_max - x_min
    y_span = y_max - y_min
    # Plot current positions for everything in the env
    for nodetype in eval_stg.env.NodeType:
        cur_nodes = scene.present_nodes(
            timestep,
            type=nodetype,
            min_history_timesteps=0,
            min_future_timesteps=0,
            return_robot=not eval_stg.hyperparams["incl_robot_node"],
        )
        if int(timestep) in cur_nodes:
            for node in cur_nodes[int(timestep)]:
                car = 1 if node.id == "ego" else 0

                pos = node.get(timestep, {"position": ["x", "y"]})[0] + np.array(
                    [x_min, y_min]
                )
                if node.type.name == "VEHICLE":
                    r_img = rotate(
                        cars[car],
                        node.get(timestep, {"heading": ["°"]})[0, 0] * 180 / np.pi,
                        reshape=True,
                    )
                    r_img = np.clip(r_img, 0, 1)
                    oi = OffsetImage(r_img, zoom=zoom, zorder=700)
                    veh_box = AnnotationBbox(oi, pos, frameon=False)
                    veh_box.zorder = 700
                    ax.add_artist(veh_box)
                    if ad_dict is not None and node in ad_dict:
                        amount_ad = ad_dict[node]
                        if amount_ad > 0.8:
                            adcolor = "r"
                        elif amount_ad < 0.2:
                            adcolor = "g"
                        else:
                            adcolor = "y"
                        if circle_ads:
                            pass
                        else:
                            ax.text(
                                pos[0] - x_span * zoom / 3,
                                pos[1] - y_span * zoom / 3,
                                str(amount_ad),
                                color=adcolor,
                                zorder=1000,
                                weight="bold",
                            )
                else:
                    circle = plt.Circle(
                        pos,
                        node_circle_size,
                        facecolor="g",
                        edgecolor="k",
                        lw=circle_edge_width,
                        zorder=3,
                    )
                    ax.add_artist(circle)
                    if ad_dict is not None and node in ad_dict:
                        amount_ad = ad_dict[node]
                        if amount_ad > 0.8:
                            if circle_ads:
                                circle = plt.Circle(
                                    pos,
                                    node_circle_size * 3.5,
                                    fill=False,
                                    edgecolor="r",
                                    lw=2.5,
                                    zorder=610,
                                )
                                ax.add_artist(circle)
                            else:
                                ax.text(
                                    pos[0] - x_span * zoom / 3,
                                    pos[1] - y_span * zoom / 3,
                                    str(amount_ad),
                                    color="r",
                                    zorder=1000,
                                    weight="bold",
                                )
                        elif amount_ad < 0.2 and not circle_ads:
                            ax.text(
                                pos[0] - x_span * zoom / 3,
                                pos[1] - y_span * zoom / 3,
                                str(amount_ad),
                                color="g",
                                zorder=1000,
                                weight="bold",
                            )


def plot_util(
    eval_stg, nusc_map, predictions, scene, ph, timestep, tau, filename, ad_dict=None
):
    # Define ROI in nuScenes Map
    x_min = scene.x_min
    x_max = scene.x_max
    y_min = scene.y_min
    y_max = scene.y_max
    zoom = zoom_scale * min(1.6 / (x_max - x_min), 1.6 / (y_max - y_min))

    my_patch = (x_min, y_min, x_max, y_max)
    fig, ax = nusc_map.render_map_patch(
        my_patch, layers, figsize=(10, 10), alpha=0.1, render_egoposes_range=False
    )
    ax.plot(
        [],
        [],
        "ko-",
        zorder=620,
        markersize=4,
        linewidth=2,
        alpha=0.7,
        label="Ours (MM)",
    )
    ax.plot(
        [],
        [],
        "w--o",
        label="Ground Truth",
        linewidth=3,
        path_effects=[pe.Stroke(linewidth=4, foreground="k"), pe.Normal()],
    )

    # Plot predictions from now until end of prediction horizon
    plot_predictions(
        ax,
        predictions,
        scene.dt,
        max_hl=10,
        ph=ph,
        map=None,
        my_patch=my_patch,
        tau=tau,
        ad_dict=ad_dict,
    )

    # Plot current positions for everything in the env
    plot_agents(ax, eval_stg, scene, timestep + tau, my_patch, zoom, ad_dict=None)

    y_span = (y_max - y_min) / 2 // zoom_scale
    y_middle = (y_max + y_min) // 2
    x_span = (x_max - x_min) / 2 // zoom_scale
    x_middle = (x_max + x_min) / 2
    ax.set_ylim((y_middle - y_span, y_middle + y_span))

    ax.set_xlim((x_middle - x_span, x_middle + x_span))
    ax.set_aspect(1)
    ax.grid(False)
    ax.axis("off")
    ax.get_legend().remove()
    fig.savefig(filename, dpi=300)
    fig.clf()

    return None
