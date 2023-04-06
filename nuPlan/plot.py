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
import os


filename_base = "./plots/imgs_for_labels_3/"

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

line_colors = ["#375397", "#F05F78", "#80CBE5", "#ABCB51", "#C8B0B0"]

line_alpha = 0.7
line_width = 0.2
edge_width = 2
circle_edge_width = 0.5
node_circle_size = 0.3
zoom_scale = 2.0

circle_ads = False


def plot_predictions(ax, predictions, my_patch, timestep, tau, ph):
    x_min, y_min, x_max, y_max = my_patch
    shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

    for node in predictions.keys():
        prediction = predictions[node]  # + shift

        future = node.get(
            np.array([timestep + 1, timestep + ph]), {"position": ["x", "y"]}
        )
        avg_prediction = np.mean(prediction[0], axis=0)
        color = "r" if node.id == "ego" else "b"

        if node.type.name == "VEHICLE":
            if not node.id == "ego":
                for t in range(tau, prediction.shape[2]):
                    sns.kdeplot(
                        prediction[0, :, t, 0],
                        prediction[0, :, t, 1],
                        ax=ax,
                        shade=True,
                        shade_lowest=False,
                        color=color,
                        zorder=600,
                        alpha=0.8,
                    )
                ax.plot(
                    avg_prediction[tau:, 0],
                    avg_prediction[tau:, 1],
                    "ko-",
                    zorder=620,
                    markersize=2,
                    linewidth=3,
                    alpha=0.7,
                )
            if not node.id == "ego":
                ax.plot(future[tau:, 0], future[tau:, 1], "r-", linewidth=2, zorder=650)
            else:
                ax.plot(future[tau:, 0], future[tau:, 1], "b-", linewidth=2, zorder=650)
        else:
            for t in range(tau, prediction.shape[2]):
                sns.kdeplot(
                    prediction[0, :, t, 0],
                    prediction[0, :, t, 1],
                    ax=ax,
                    shade=True,
                    shade_lowest=False,
                    color=color,
                    zorder=600,
                    alpha=0.8,
                )

            ax.plot(
                avg_prediction[tau:, 0],
                avg_prediction[tau:, 1],
                "ko-",
                zorder=620,
                markersize=2,
                linewidth=1,
                alpha=0.7,
            )
            ax.plot(
                future[tau:, 0],
                future[tau:, 1],
                "g-",
                zorder=650,
                path_effects=[
                    pe.Stroke(linewidth=edge_width, foreground="k"),
                    pe.Normal(),
                ],
            )


def new_plot(name, scene, predictions, timestep, ph):
    name = name[:6] + "/"
    if not os.path.exists(filename_base + name):
        os.makedirs(filename_base + name)

    x_min = scene.x_min
    x_max = scene.x_max
    y_min = scene.y_min
    y_max = scene.y_max

    my_patch = (x_min, y_min, x_max, y_max)
    for t in [0]:  # range(ph):
        fig, ax = plt.subplots(1, 1)

        plot_predictions(ax, predictions, my_patch, timestep, t, ph)

        ax.axis("off")
        filename = filename_base + name + str(timestep) + ".png"
        ax.set_ylim(-150, 150)
        ax.set_xlim(-150, 150)
        fig.savefig(filename, dpi=300)
        fig.clf()


def new_plot2(name, scene, predictions, timestep, ph):
    name = name + "/"
    if not os.path.exists(filename_base + name):
        os.makedirs(filename_base + name)

    x_min = scene.x_min
    x_max = scene.x_max
    y_min = scene.y_min
    y_max = scene.y_max

    my_patch = (x_min, y_min, x_max, y_max)
    for t in [0]:  # range(ph):
        fig, ax = plt.subplots(1, 1)

        plot_predictions(ax, predictions, my_patch, timestep, t, ph)

        ax.axis("off")
        filename = filename_base + name + str(timestep) + ".png"
        ax.set_ylim(0, 300)
        ax.set_xlim(0, 300)
        fig.savefig(filename, dpi=300)
        fig.clf()
