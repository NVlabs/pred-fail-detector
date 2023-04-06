import random
import string
import pathlib

from nuplan.common.actor_state.vehicle_parameters import (
    VehicleParameters,
    get_pacifica_parameters,
)
from nuplan.planning.metrics.evaluation_metrics.common.ego_is_comfortable import (
    EgoIsComfortableStatistics,
)
from nuplan.planning.metrics.abstract_metric import AbstractMetricBuilder

example_metric: AbstractMetricBuilder = EgoIsComfortableStatistics(
    name="ego_is_comfortable_statistics",
    category="Violations",
    min_lon_accel=-4.05,
    max_lon_accel=2.4,
    max_abs_lat_accel=4.89,
    max_abs_yaw_rate=0.95,
    max_abs_yaw_accel=1.93,
    max_abs_lon_jerk=4.13,
    max_abs_mag_jerk=8.37,
)

VEHICLE_PARAMETERS: VehicleParameters = get_pacifica_parameters()

cur_dir = str(pathlib.Path(__file__).parent.resolve())

DATA_ROOT = cur_dir+"/dataset"
DATA_VERSION = "nuplan_v0.2_mini"
# DATA_VERSION = 'nuplan_v0.2'

SAVE_DIR = "saves/"
SIMULATION_DIR = "simulation"
METRIC_DIR = "metrics"


def generate_token(n=32):
    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(n)
    )
