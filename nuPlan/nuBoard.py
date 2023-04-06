from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from override.nuboard.nuboard import NuBoard

# from nuplan.planning.nuboard.nuboard import NuBoard as SimBoard
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan_utils import VEHICLE_PARAMETERS, DATA_ROOT, SAVE_DIR, DATA_VERSION

EXPERIMENT = "predictorplanner"
# Fetch the filesystem location of the simulation results file for visualization in nuBoard (next section)
parent_dir = Path(SAVE_DIR) / EXPERIMENT
results_dir = list(parent_dir.iterdir())[-1]
print(results_dir)
# exit()
nuboard_file = [
    str(file)
    for file in results_dir.iterdir()
    if file.is_file() and file.suffix == ".nuboard"
][0]
scenario_builder = NuPlanScenarioBuilder(DATA_VERSION, DATA_ROOT)
metric_categories = ["Dynamics", "Planning", "Scenario dependent", "Violations"]

frame_rate = 4  # 2 Hz default
nuboard = NuBoard(
    nuboard_paths=[nuboard_file],
    scenario_builder=scenario_builder,
    metric_categories=metric_categories,
    vehicle_parameters=VEHICLE_PARAMETERS,
    frame_rate=frame_rate,
)

nuboard.run()
