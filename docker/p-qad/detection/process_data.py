import numpy as np
import sys

sys.path.append("../")

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from anomaly_detector import anomaly_detection_all

exp_data_folder = "../saves/"


AD_SUM = 0
AD_D2A = 1
MEAN = 2
UDT = 3
PDT = 4
TTC = 5
LIKELIHOOD = 6
HJ = 7
ENSEMBLE = (0, 3, 4)
symbol = ["o", "s", "*", "^", "v", "D", "p", ">"]
lines = ["b-", "g-", "r-", "m-", "y-", "c-"]
colors = ["red", "limegreen", "blue", "magenta", "gold", "darkorange", "red", "red"]
ad_range = [1.01] + [1 - 0.01 * i for i in range(101)]
bfar_range = ad_range
likelihood_range = np.sort(
    [0] + [1e-15 * (2**i) for i in range(60)] + [np.inf] + [0.048, 0.05, 0.055]
)
ttc_range = [-10, -5, -1] + [0.01 * i**2 for i in range(72)]
hj_reachability_range = (
    [-2 + 0.1 * i for i in range(20)] + [0.5 * i for i in range(30)] + [np.inf]
)


def characterize_adaptive(ads):
    from nuPlan.labels import (
        names_ego_at_pudo,
        names_ego_following_vehicle,
        names_ego_stopping_at_traffic_light,
        names_nearby_dense_vehicle_traffic,
    )

    ph = 20
    dt = 0.5
    threshold = 0.75
    namess = [
        names_ego_at_pudo,
        names_ego_following_vehicle,
        names_ego_stopping_at_traffic_light,
        names_nearby_dense_vehicle_traffic,
    ]
    legends = [
        "Ego at pick up/drop off       ",
        "Ego following vehicle        ",
        "Ego stopping at traffic light",
        "Nearby dense vehicle traffic ",
    ]
    results = [[], [], [], []]
    nums = [[], [], [], []]
    for name in ads.keys():
        ad = np.array(ads[name][0])[:ph]
        num = ads[name][1]

        ad = convert_to_boolean(ad, value=threshold, operation="geq")
        t_a = np.where(ad)[0]
        t_a = ph * dt if len(t_a) == 0 else t_a[0] * dt
        # if t_a == 10 or t_a == 0 or t_a == 0.5:
        # continue
        for i, (names, result) in enumerate(zip(namess, results)):
            if name in names:
                nums[i].append(num)
                result.append(t_a)
                break

    fig = plt.figure()
    fig.set_figwidth(6)
    fig.set_figheight(3)
    for i, (legend, result) in enumerate(zip(legends, results)):
        m = np.mean(result)
        s = np.std(result)
        l = len(result)
        print(legend, m, "pm", s, ": len", l)
        x, h = np.histogram(result, bins=20)
        x = [0, *(np.cumsum(x) / len(result))]
        plt.plot([0.5 * i for i in range(0, 21)], x, label=legend)

    plt.legend(loc="lower right")
    plt.xlabel("Time (s)")
    plt.ylabel("Proportion of Detections")
    plt.tight_layout()
    plt.savefig("plots/adaptive.png", dpi=200)


def save_new_ads(ads, m, start):
    ads2 = {}
    folder = exp_data_folder + "subm500/"
    filename = "m" + str(m) + "s" + str(start) + ".npy"

    for name in ads:
        print(name)
        phs = ads[name]
        ads2[name] = []
        for ph in phs:
            preds = ph[1]
            achs = ph[2]
            ad = aggregate_anomaly_detection_sub(4, preds, achs, m=m, start=start)
            ads2[name].append(ad)
        ads2[name] = np.array(ads2[name])

    np.save(folder + filename, ads2)


def print_info(ads):
    ads = np.array(ads)
    print("Number of prediction horizons:", len(ads))
    print("Number of detections:")
    a = []
    a.append(ads[:, 0] >= 0.99)
    a.append(ads[:, 0] >= 0.91)
    a.append(ads[:, 3] <= 0.05)
    a.append(ads[:, 4] <= 0.05)
    a.append(ads[:, 5] <= 1.0)
    a.append(ads[:, 6] <= 0.05)
    a.append(ads[:, 7] < 0)

    print("Q-ADfpr:", np.sum(a[0]) / len(ads))
    print("Q-ADfnr:", np.sum(a[1]) / len(ads))
    print("UDT:", np.sum(a[2]) / len(ads))
    print("PDT:", np.sum(a[3]) / len(ads))
    print("TTC:", np.sum(a[4]) / len(ads))
    print("LIKELIHOOD:", np.sum(a[5]) / len(ads))
    print("HJ:", np.sum(a[6]) / len(ads))

    for y in a:
        v = np.sum(np.bitwise_and(a[0], y)) / sum(a[0])
        print(v)


def print_positives(ads, column, threshold, operation):

    for key in ads.keys():
        ad_result = convert_to_boolean(ads[key][:, column], threshold, operation)
        for i, boolean in enumerate(ad_result):
            if boolean:
                print(key, i)


def serialize_data(data):
    serialized_data = None

    for key in data.keys():
        if serialized_data is None:
            serialized_data = data[key]
        else:
            serialized_data = np.concatenate((serialized_data, data[key]), axis=0)

    return serialized_data


def count_tptnfpfn(xss, ys, multi=False):
    if not multi:
        xss = [xss]
    tptnfpfns = []

    for xs in xss:
        xs = xs[:16]  # only labeling at most first 8 steps
        tptnfpfn = [0, 0, 0, 0]
        for x, y in zip(xs, ys):
            if y == 0:  # negative
                if x:  # false positive
                    tptnfpfn[2] += 1
                else:  # true negative
                    tptnfpfn[1] += 1
            else:  # positive
                if x:  # true positive
                    tptnfpfn[0] += 1
                else:  # false negative
                    tptnfpfn[3] += 1
        tptnfpfns.append(tptnfpfn)
    if not multi:
        tptnfpfns = tptnfpfns[0]

    return tptnfpfns


def convert_to_boolean(data, value=0.99, operation="leq", multi=False):
    if operation == "l":
        return data < value
    elif operation == "g":
        return data > value
    elif operation == "leq":
        return data <= value
    elif operation == "geq":
        return data >= value
    elif operation == "eq":
        return data == value
    else:
        raise NotImplementedError


def count_all_tptnfpfn(data, labels, column, threshold, operation, multi=False):
    tptnfpfn = 0  # np.array([0,0,0,0])
    if type(column) == int:
        for key in data.keys():
            if multi:
                data_col = data[key][:, :, column]
            else:
                data_col = data[key][:, column]
            result = convert_to_boolean(data_col, threshold, operation, multi=multi)
            tptnfpfn += np.array(count_tptnfpfn(result, labels[str(key)], multi=multi))
    elif type(column) == tuple:
        print(column, threshold, operation)
        typ = column[0]
        column = column[1:]
        for key in data.keys():

            if typ == "fpr":
                if multi:
                    ensemb_result = np.ones((data[key].shape[0], data[key].shape[2]))
                else:
                    ensemb_result = np.ones((len(data[key][:, 0]),))
            elif typ == "fnr":
                if multi:
                    ensemb_result = np.zeros((data[key].shape[0], data[key].shape[2]))
                else:
                    ensemb_result = np.zeros((len(data[key][:, 0]),))
            else:
                raise NotImplementedError

            for i in range(len(column)):
                if multi:
                    data_col = data[key][:, :, column[i]]
                else:
                    data_col = data[key][:, column[i]]

                result = convert_to_boolean(
                    data_col, threshold[i], operation[i], multi=multi
                )
                if typ == "fpr":
                    ensemb_result = np.minimum(ensemb_result, result[:, :8])
                elif typ == "fnr":
                    ensemb_result = np.maximum(ensemb_result, result[:, :8])
                else:
                    raise NotImplementedError
            tptnfpfn += np.array(
                count_tptnfpfn(ensemb_result, labels[str(key)], multi=multi)
            )

    return tptnfpfn


def get_np(tptnfpfn, multi=False):
    if multi:
        p = tptnfpfn[:, 0] + tptnfpfn[:, 3]
        n = tptnfpfn[:, 1] + tptnfpfn[:, 2]
    else:
        p = tptnfpfn[0] + tptnfpfn[3]
        n = tptnfpfn[1] + tptnfpfn[2]
    return n, p


def auc(xs, ys):
    inds = np.argsort(xs)
    xs = np.array(xs)[inds]
    ys = np.array(ys)[inds]
    area = 0
    for i in range(len(xs) - 1):
        w = xs[i + 1] - xs[i]
        area += w * (ys[i] + ys[i + 1]) / 2

    return area


def FPR_TPR(data, labels, column, threshold, operation, returns=None, multi=False):
    tptnfpfn = count_all_tptnfpfn(
        data, labels, column, threshold, operation, multi=multi
    )
    n, p = get_np(tptnfpfn, multi=multi)
    if multi:
        fpr = tptnfpfn[:, 2] / n
        tpr = tptnfpfn[:, 0] / p
        fnr = tptnfpfn[:, 3] / p
        tnr = tptnfpfn[:, 1] / n
    else:
        fpr = tptnfpfn[2] / n
        tpr = tptnfpfn[0] / p
        fnr = tptnfpfn[3] / p
        tnr = tptnfpfn[1] / n
    if returns is None:
        return (
            fpr,
            tpr,
        )
    elif returns == "all":
        return fpr, tpr, fnr, tnr


def find_best_on_roc(xs, ys, thresholds):
    max_dist = 0
    best_ind = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        dist = (-x + y) / np.sqrt(2)
        if dist > max_dist:
            max_dist = dist
            best_ind = i
    return thresholds[best_ind]


def roc_plot(ads, labels, thresholds, exp_name="", multi=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    fig.set_figwidth(4)
    fig.set_figheight(4)
    methods = [
        AD_SUM,
        UDT,
        PDT,
        LIKELIHOOD,
        TTC,
        HJ,
        AD_SUM,
        ("fpr", AD_SUM, UDT, PDT, HJ),
        ("fnr", AD_SUM, UDT, PDT, HJ),
    ]
    operations = [
        "geq",
        "leq",
        "leq",
        "leq",
        "leq",
        "leq",
        "geq",
        ("geq", "leq", "leq", "leq"),
        ("geq", "leq", "leq", "leq"),
    ]
    ranges = [
        ad_range,
        bfar_range,
        bfar_range,
        likelihood_range,
        ttc_range,
        hj_reachability_range,
    ]
    legends = [
        "QAD",
        "UDT",
        "PDT",
        "Likelihood",
        "TTC",
        "HJ Reachability",
        "QAD",
        "FPR Ensemble",
        "FNR Ensemble",
    ]

    symbol = ["o", "s", "*", "", "", "D", ">", "^", "<"]
    colors = [
        "red",
        "limegreen",
        "blue",
        "magenta",
        "gold",
        "darkorange",
        "red",
        "black",
        "black",
    ]

    for i in range(len(ranges)):
        xs = []
        ys = []
        for m in ranges[i]:
            fpr, tpr = FPR_TPR(ads, labels, methods[i], m, operations[i], multi=multi)
            xs.append(fpr)
            ys.append(tpr)
        if multi:
            xs = np.array(xs).T
            ys = np.array(ys).T
            fs = []
            best_on_rocs = []
            for x, y in zip(xs, ys):
                best_on_rocs.append(find_best_on_roc(x, y, ranges[i]))
                if x[-1] != 0 and x[-1] != 1:
                    x = [*x, round(x[-1])]
                    y = [*y, round(y[-1])]
                elif x[0] != 0 and x[0] != 1:
                    x = [round(x[0]), *x]
                    y = [round(y[0]), *y]
                fs.append(interp1d(x, y))
            xs = np.arange(0, 1.001, 0.001)
            ys = []
            for f in fs:
                ys.append(f(xs))
            ys = np.array(ys)
            areas = [auc(xs, ys[j]) for j in range(len(ys))]
            print(
                legends[i],
                np.round(np.mean(areas), 3),
                "pm",
                np.round(np.std(areas), 3),
            )
            print("best on roc", best_on_rocs, np.mean(best_on_rocs))

            mean = np.mean(ys, axis=0)
            std = np.std(ys, axis=0)
            plt.plot(xs, mean, lines[i], color=colors[i], label=legends[i])
            plt.fill_between(
                xs, mean, mean + 2 * std, color=colors[i], alpha=0.3, linewidth=0.0
            )
            plt.fill_between(
                xs, mean, mean - 2 * std, color=colors[i], alpha=0.3, linewidth=0.0
            )

        else:
            print(legends[i], auc(xs, ys))
            # print(find_best_on_roc(xs,ys, ranges[i]))
            plt.plot(xs, ys, lines[i], color=colors[i], label=legends[i])

    for i in range(len(methods)):
        fpr, tpr, fnr, _ = FPR_TPR(
            ads, labels, methods[i], thresholds[i], operations[i], "all", multi=multi
        )
        if multi:
            dist = (-fpr + tpr) / np.sqrt(2)
            print(
                legends[i % len(legends)],
                np.round(100 * np.mean(fpr), 1),
                "pm",
                np.round(100 * np.std(fpr), 1),
                ";",
                np.round(100 * np.mean(fnr), 1),
                "pm",
                np.round(100 * np.std(fnr), 1),
                ": dist",
                np.round(np.mean(dist), 3),
                "pm",
                np.round(np.std(dist), 3),
            )

            plt.plot(np.mean(fpr), np.mean(tpr), symbol[i], color=colors[i], alpha=1)
        else:
            print(legends[i % len(legends)], fpr, fnr, fpr + fnr)
            plt.plot(fpr, tpr, symbol[i], color=colors[i], alpha=1)

    plt.plot([0.1 * i for i in range(11)], [0.1 * i for i in range(11)], "k")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("plots/roc1" + exp_name + ".png", dpi=300)

def get_adss_nuPlan_reactive(num_seeds):
    expname = "nuPlan_reactive_"
    expname2 = "nuPlan_reactive2_"
    adss = []
    for i in range(num_seeds):
        ads1 = (
            np.load(exp_data_folder + expname + str(i) + ".npy", allow_pickle=True)
            .item()
            .get("ads")
        )
        ads2 = (
            np.load(exp_data_folder + expname2 + str(i) + ".npy", allow_pickle=True)
            .item()
            .get("ads")
        )
        adss.append({**ads1, **ads2})

    adss_compiled = {}
    for name in adss[0]:
        adss_compiled[name] = [[] for _ in range(len(adss))]
        ads_temp = []
        for i, ads in enumerate(adss):
            phs = ads[name]
            if i in [0, 1, 2, 3, 4]:
                adss_compiled[name][i] = phs
            else:
                adss_compiled[name][i] = np.array([ph[0] for ph in phs])

        adss_compiled[name] = np.array(adss_compiled[name])

        return adss_compiled


if __name__=="__main__":

    import argparse
    from tqdm import tqdm

    def collect_as(coll_type):
        class Collect_as(argparse.Action):
            def __call__(self, parser, namespace, values, options_string=None):
                setattr(namespace, self.dest, coll_type(values))

        return Collect_as

    parser = argparse.ArgumentParser(description="Compute Cost Matrix")
    parser.add_argument("--exp", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=1)

    args = parser.parse_args()
    exp_num = args.exp
    num_seeds = args.num_seeds

    if exp_num == 1:
        exp_name = "fixed plan (nuScenes and nuPlan)"
        exp1_name = "nuScenes_all"
        exp2_name = "nuPlan_fixed_plan"

        for i in range(num_seeds):
            data_dict = np.load(exp_data_folder + exp1_name + "_" + str(i) + ".npy", allow_pickle=True).item()
            ads1 = data_dict.get("ads")
            data_dict = np.load(exp_data_folder + exp2_name + "_" + str(i) + ".npy", allow_pickle=True).item()
            ads2 = data_dict.get("ads")

            ads_dict = {**ads1, **ads2}
            ads = serialize_data(ads_dict)
            print("\nSeed:", i)
            print_info(ads)

    elif exp_num == 2:
        exp_name = "nuPlan_adaptive"
        data_dict = np.load(exp_data_folder + exp_name + ".npy", allow_pickle=True).item()
        ads = data_dict.get("ads")
        characterize_adaptive(ads)

    elif exp_num == 3:
        expname = "nuPlan_reactive_"
        expname2 = "nuPlan_reactive2_"
        adss = []
        for i in range(num_seeds):
            ads1 = (
                np.load(exp_data_folder + expname + str(i) + ".npy", allow_pickle=True)
                .item()
                .get("ads")
            )
            ads2 = (
                np.load(exp_data_folder + expname2 + str(i) + ".npy", allow_pickle=True)
                .item()
                .get("ads")
            )
            adss.append({**ads1, **ads2})
        
        adss_compiled = {}
        for name in adss[0]:
            adss_compiled[name] = [[] for _ in range(len(adss))]
            ads_temp = []
            for i, ads in enumerate(adss):
                phs = ads[name]
                if i in [0, 1, 2, 3, 4]:
                    adss_compiled[name][i] = phs
                else:
                    adss_compiled[name][i] = np.array([ph[0] for ph in phs])

            adss_compiled[name] = np.array(adss_compiled[name])

        from nuPlan.labels import labels

        thresholds = [
            0.91,
            0.05,
            0.05,
            0.05,
            1,
            0,
            0.99,
            (0.99, 0.05, 0.05, 0),
            (0.91, 0.05, 0.05, 0),
        ]
        roc_plot(adss_compiled, labels, thresholds, "nuPlan_reactive_multi", multi=True)

    elif exp_num == 4:
        expname = "nuPlan_reactive_"
        expname2 = "nuPlan_reactive2_"
        adss = []
        for i in range(num_seeds):
            ads1 = (
                np.load(exp_data_folder + expname + str(i) + ".npy", allow_pickle=True)
                .item()
                .get("ads")
            )
            ads2 = (
                np.load(exp_data_folder + expname2 + str(i) + ".npy", allow_pickle=True)
                .item()
                .get("ads")
            )
            adss.append({**ads1, **ads2})

        adss_compiled = {}
        for name in adss[0]:
            adss_compiled[name] = [[] for _ in range(len(adss))]
            ads_temp = []
            for i, ads in enumerate(adss):
                phs = ads[name]
                if i in [0, 1, 2, 3, 4]:
                    adss_compiled[name][i] = phs
                else:
                    adss_compiled[name][i] = np.array([ph[0] for ph in phs])

            adss_compiled[name] = np.array(adss_compiled[name])

        from nuPlan.labels import labels

        thresholds = [
            0.56, 
            0.98, 
            0.838, 
            0.0703, 
            1, 
            1.5, 
            0.99, 
            (0.56, 0.98, 0.87,1.5), 
            (0.56, 0.98, 0.87, 1.5),
        ]
        roc_plot(adss_compiled, labels, thresholds, "nuPlan_reactive_multi", multi=True)


    # serialize_data(ads)
    # qs =     [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30]  # quantiles we look at for detection
    # fpr_ms = [1.01, 1.01, 1.00, 0.99, 0.96, 0.87, 0.78]  # required threshold for at most 0.05 fpr guarantee
    # fpr_gt = [0,    0,    0.05, 0.04, 0.02, 0.05, 0.05]  # guarantee associated with fpr, m
    # fnr_ms = [0.98, 0.96, 0.94, 0.91, 0.85, 0.73, 0.62]  # required threshold for at most 0.05 fnr guarantee
    # fnr_gt = [0.02, 0.05, 0.03, 0.03, 0.04, 0.03, 0.03]  # guarantee associated with fpr, m
