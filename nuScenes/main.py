import sys

sys.path.append("../trajectron")
sys.path.append("../")
import os
import numpy as np
import torch
import dill
import warnings

warnings.filterwarnings("ignore")
import imageio as io
from model.model_registrar import ModelRegistrar
from model import Trajectron
import json
from plot import plot_util
from detection.cost_util import CostUtil
from detection.anomaly_detector import aggregate_anomaly_detection, anomaly_detection

from nuPlan.plot import new_plot2

from nuscenes.map_expansion.map_api import NuScenesMap


def load_model(model_dir, env, ts):
    model_registrar = ModelRegistrar(model_dir, "cpu")
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, "config.json"), "r") as config_json:
        hyperparams = json.load(config_json)

    hyperparams["map_enc_dropout"] = 0.0
    if "incl_robot_node" not in hyperparams:
        hyperparams["incl_robot_node"] = False

    stg = Trajectron(model_registrar, hyperparams, None, "cpu")

    stg.set_environment(env)

    stg.set_annealing_params()

    return stg, hyperparams


def main(i, step=None):
    advalues = []
    scene = eval_scenes[i]
    ego_node = scene.robot
    criterion = CostUtil(scene)
    timesteps = scene.timesteps
    nusc_map = NuScenesMap(dataroot="./", map_name=scene.map_name)

    ph = 4
    predict_num_samples = 100
    plotting = False
    plotting_for_label = False

    ad = True
    from labels import labels

    filename_base = "plots/gif_imgs/" + str(i)
    gifname = "plots/val" + str(i) + ".gif"
    filenames = []
    crop = (timesteps // ph - 1) * ph + 2
    start = 2
    if step is not None:
        start += 4 * step
    for timestep in range(start, crop, ph):  # range(2,crop,ph):
        timestep = np.array([timestep])
        pred_timesteps = np.array([int(timestep) + 1, int(timestep) + ph])

        with torch.no_grad():
            predictions = eval_stg.predict(
                scene, timestep, ph, num_samples=predict_num_samples
            )
            dists, _ = eval_stg.predict(
                scene, timestep, ph, num_samples=1, output_dists=True, gmm_mode=True
            )

        ads = aggregate_anomaly_detection(
            criterion,
            ego_node,
            ph,
            pred_timesteps,
            predictions[int(timestep)],
            dists[int(timestep)],
        )
        advalues.append(ads)

        if plotting:
            predicted_cost = criterion.compute_noplan_cost(
                ego_node, pred_timesteps, predictions[int(timestep)], split_agents=True
            )
            for tau in [0]:  # range(ph):
                achieved_cost = criterion.compute_noplan_cost(
                    ego_node,
                    np.array([int(timestep + tau + 1), int(timestep + tau + 1)]),
                    split_agents=True,
                )
                ad_dict = anomaly_detection(
                    achieved_cost, predicted_cost, tau, return_avg_diff=False
                )
                filename = filename_base + str(int(timestep)) + "+" + str(tau) + ".png"
                filenames.append(filename)
                print(i, timestep, ad_dict["all"])
                print(ad_dict)
                plot_util(
                    eval_stg,
                    nusc_map,
                    predictions,
                    scene,
                    ph,
                    timestep,
                    tau + 1,
                    filename,
                    ad_dict=ad_dict if ad else None,
                )
                return

        if plotting_for_label:
            print("Evaluating...")
            new_plot2(str(i), scene, predictions[int(timestep)], int(timestep), ph)

    data_dict["ads"][i] = np.array(advalues)


if __name__ == "__main__":

    import argparse
    from tqdm import tqdm

    def collect_as(coll_type):
        class Collect_as(argparse.Action):
            def __call__(self, parser, namespace, values, options_string=None):
                setattr(namespace, self.dest, coll_type(values))

        return Collect_as

    parser = argparse.ArgumentParser(description="Compute Cost Matrix")
    parser.add_argument("--exp", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    exp = args.exp
    seed = args.seed

    exp_data_folder = "../saves/"
    log_dir = "../models"
    model_dir = os.path.join(log_dir, "int_ee")

    if exp == 0:
        with open("./processed/nuScenes_val.pkl", "rb") as f:
            eval_env = dill.load(f, encoding="latin1")
    elif exp == 1:
        exp_name = "nuScenes_all_"+str(seed)
        with open("./processed/nuScenes_val.pkl", "rb") as f:
            eval_env = dill.load(f, encoding="latin1")
    elif exp == 2:
        exp_name = "nuScenes_training_"+str(seed)
        with open("./processed/nuScenes_train.pkl", "rb") as f:
            eval_env = dill.load(f, encoding="latin1")
    else:
        raise NotImplementedError

    eval_scenes = eval_env.scenes
    eval_stg, _ = load_model(model_dir, eval_env, ts=12)

    data_dict = {"ads": {}}

    if exp > 0:
        num_scenes = len(eval_scenes)
        iis = range(num_scenes)
        for i in tqdm(iis):
            main(i)
        np.save(exp_data_folder + exp_name + ".npy", data_dict)
    else:
        # visuals
        iis = [41]
        steps = [None]

        for i, step in zip(iis, steps):
            main(i, step)
