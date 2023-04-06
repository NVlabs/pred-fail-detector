import numpy as np
from drd.StatsCalculator import SimpleMean
from drd.StatsCalculator import Hotelling
from drd.StatsCalculator import WeightedMean as UDT
from drd.StatsCalculator import TransformedCVaR as PDT
import time


def anomaly_detection(
    achieved_cost_dict, predicted_cost_dict, tau, return_avg_diff=False
):
    ad_dict = {}
    predicted_costs = 0
    achieved_costs = 0
    for node in achieved_cost_dict.keys():
        if (
            node in predicted_cost_dict
        ):  # exclude other cost terms which we won't do AD on
            predicted_cost = predicted_cost_dict[node][:, tau]
            achieved_cost = achieved_cost_dict[node]
            predicted_costs += predicted_cost
            achieved_costs += achieved_cost
            rank_proportion = np.mean(predicted_cost < achieved_cost)
            if return_avg_diff:
                diff = float(np.round(achieved_cost - np.mean(predicted_cost), 4))
                ad_dict[node] = (rank_proportion, diff)
            else:
                ad_dict[node] = rank_proportion

    rank_proportion = np.mean(predicted_costs < achieved_costs)
    if return_avg_diff:
        diff = float(np.round(achieved_costs - np.mean(predicted_costs), 4))
        ad_dict["all"] = (rank_proportion, diff)
    else:
        ad_dict["all"] = rank_proportion

    return ad_dict


def anomaly_detection_all(achieved_cost, predicted_cost, tau):
    predicted_cost = predicted_cost[:, tau]
    rank_proportion = np.mean(predicted_cost < achieved_cost)
    return rank_proportion


def likelihood_detection(agent_likelihoods, tau, likelihood_threshold=0.01):
    ld_dict = {}

    any_detection = 0
    for node in agent_likelihoods.keys():
        likelihood = agent_likelihoods[node][tau]

        detection = int(likelihood < likelihood_threshold)
        if detection:
            any_detection = 1

        ld_dict[node] = detection

    ld_dict["all"] = any_detection

    return ld_dict


def likelihood_detection_all(min_agent_likelihood, tau, likelihood_threshold=0.01):
    detection = int(min_agent_likelihood[0, tau] < likelihood_threshold)

    return detection


def aggregate_anomaly_detection(
    criterion,
    ego_node,
    ph,
    prediction_timesteps,
    predictions,
    dists,
    likelihood_threshold=0.05,
):

    predicted_cost = criterion.compute_noplan_cost(
        ego_node, prediction_timesteps, predictions
    )
    predicted_d2a_cost = criterion.compute_noplan_distance_to_agent_cost(
        ego_node, prediction_timesteps, predictions
    )
    agent_likelihoods = criterion.compute_likelihood_cost(
        ego_node, prediction_timesteps, dists
    )
    simplemean = SimpleMean(X=predicted_cost, side=1)
    udt = None
    pdt = None
    try:
        udt = UDT(X=predicted_cost, side=1)
    except:
        print("udt issue")
    try:
        pdt = PDT(X=-1 * predicted_cost, p=0.5)
    except:
        print("pdt issue")

    ad1 = [0, 0]
    ad2 = [1, 1, 1, 50, np.inf, np.inf]
    achieved_cost = criterion.compute_noplan_cost(
        ego_node, timesteps=prediction_timesteps
    )
    achieved_d2a_cost = criterion.compute_noplan_distance_to_agent_cost(
        ego_node, timesteps=prediction_timesteps
    )
    achieved_ttc = criterion.compute_time_to_collision(
        ego_node, timesteps=prediction_timesteps
    )
    achieved_target_values = criterion.compute_target_value(
        ego_node, timesteps=prediction_timesteps
    )

    for tau in range(0, ph):
        adt = [
            anomaly_detection_all(achieved_cost[:, tau : tau + 1], predicted_cost, tau),
            anomaly_detection_all(
                achieved_d2a_cost[:, tau : tau + 1], predicted_d2a_cost, tau
            ),
        ]
        ad1 = np.maximum(adt, ad1)

        adt = [
            simplemean.bootstrap_main(achieved_cost[0, : tau + 1], side=1)[0],
            udt.bootstrap_main(achieved_cost[0, : tau + 1], side=1)[0]
            if udt is not None
            else 1,
            pdt.bootstrap_main(-1 * achieved_cost[0, : tau + 1])[0]
            if pdt is not None
            else 1,
            achieved_ttc[0, tau],
            agent_likelihoods[0, tau],
            achieved_target_values[0, tau],
        ]
        ad2 = np.minimum(adt, ad2)

    ad_values = [*ad1, *ad2]

    return ad_values


def aggregate_adaptive_planning(
    criterion,
    ego_node,
    ph,
    prediction_timesteps,
    predictions,
    dists,
    likelihood_threshold=0.05,
):
    predicted_cost = criterion.compute_noplan_cost(
        ego_node, prediction_timesteps, predictions
    )
    achieved_cost = criterion.compute_noplan_cost(
        ego_node, timesteps=prediction_timesteps
    )

    adt = []
    for tau in range(0, ph):
        adt.append(
            anomaly_detection_all(achieved_cost[:, tau : tau + 1], predicted_cost, tau)
        )

    return adt
