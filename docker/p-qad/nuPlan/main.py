import pathlib, logging, time
import pytorch_lightning as pl
import numpy as np
from nuplan.planning.script.default_path import set_default_path
from nuplan.planning.simulation.callback.metric_callback import MetricCallBack
from nuplan.planning.simulation.simulation import Simulation, SimulationSetup
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.utils.multithreading.worker_ray import RayDistributed
from nuplan.planning.simulation.callback.serialization_callback import (
    SerializationCallback,
)
from nuplan.planning.simulation.controller.perfect_tracking import (
    PerfectTrackingController,
)
from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.simulation_manager.step_simulation_manager import (
    StepSimulationManager,
)

# from nuplan.planning.simulation.observation.idm_agents_observation import IDMAgentsObservation
from override.observation.idm_agents_observation import IDMAgentsObservation
from nuplan.planning.simulation.observation.box import BoxObservation
from nuplan.planning.nuboard.base.data_class import NuBoardFile
from nuplan.planning.metrics.metric_engine import MetricsEngine
from planner import PredictorPlanner

# warnings.filterwarnings("ignore")
from nuplan_utils import (
    example_metric,
    DATA_ROOT,
    DATA_VERSION,
    SAVE_DIR,
    METRIC_DIR,
    SIMULATION_DIR,
)

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

    set_default_path()
    logger = logging.getLogger(__name__)
    logger.disabled = 1
    pl.seed_everything(0, workers=True)

    exp_data_folder = "../saves/"

    # # Options
    scenario_types = [
        "ego_accelerating_at_traffic_light",
        "ego_at_pudo",
        "ego_lane_change",
        "ego_starts_high_speed_turn",
        "ego_starts_left_turn",
        "ego_starts_protected_cross_turn",
        "ego_starts_right_turn",
        "nearby_dense_vehicle_traffic",
        "nearby_pedestrian_at_pudo",
        "ego_following_vehicle",
        "ego_high_curvature",
        "ego_stopping_at_traffic_light",
        "ego_starts_unprotected_cross_turn",
    ]

    if exp == 0:
        exp_name = "nuPlan_test_"+str(seed)
        scenario_types = ["ego_accelerating_at_traffic_light"]
        limit_scenarios_per_type = 1
        check_labels = False
        reactive_agents = True
        follow_expert = not reactive_agents
        names = []
    elif exp == 1:
        exp_name = "nuPlan_reactive_"+str(seed)
        scenario_types = [
            "ego_at_pudo",
            "nearby_dense_vehicle_traffic",
            "ego_starts_protected_cross_turn",
            "ego_accelerating_at_traffic_light",
            "ego_starts_unprotected_cross_turn",
            "ego_starts_right_turn",
        ]
        from labels import labels as scene_labels

        check_labels = False
        limit_scenarios_per_type = 125
        reactive_agents = True
        follow_expert = not reactive_agents
        names = []
    elif exp == 2:
        exp_name = "nuPlan_fixed_plan_"+str(seed)
        limit_scenarios_per_type = None
        reactive_agents = False
        follow_expert = not reactive_agents
        check_labels = False
        names = []
    elif exp == 3:
        exp_name = "nuPlan_reactive2_"+str(seed)
        scenario_types = [
            "nearby_dense_vehicle_traffic",
            "ego_accelerating_at_traffic_light",
        ]
        check_labels = False
        from labels import labels_12 as scene_labels

        names = list(scene_labels.keys())
        limit_scenarios_per_type = 50
        reactive_agents = True
        follow_expert = not reactive_agents
        DATA_VERSION = "nuplan_v0.2"
    elif exp == 4:
        exp_name = "nuPlan_adaptive_"+str(seed)
        scenario_types = [
            "ego_stopping_at_traffic_light",
            "nearby_dense_vehicle_traffic",
            "ego_at_pudo",
            "ego_following_vehicle",
        ]
        check_labels = False
        limit_scenarios_per_type = 200
        names = ["34bd51"]
        DATA_VERSION = "nuplan_v0.2"
        reactive_agents = True
        follow_expert = True
    else:
        raise NotImplementedError

    subsample_ratio = 0.1
    simulation_history_buffer_duration = 2.0

    ## File setup
    EXPERIMENT = "predictorplanner"
    EXPERIMENT_TIME = time.strftime("%Y.%m.%d.%H.%M.%S")
    output_dir = SAVE_DIR + "/" + EXPERIMENT + "/" + EXPERIMENT_TIME
    OUTPUT_DIR = pathlib.Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_PATH = OUTPUT_DIR / METRIC_DIR

    ## Building nuboard event file.
    nuboard_filename = OUTPUT_DIR / (
        f"nuboard_{int(time.time())}" + NuBoardFile.extension()
    )
    nuboard_file = NuBoardFile(
        main_path=output_dir, simulation_folder=SIMULATION_DIR, metric_folder=METRIC_DIR
    )
    nuboard_file.save_nuboard_file(file=nuboard_filename)

    ## Scenario Selection
    filter = ScenarioFilters(
        shuffle=False,
        flatten_scenarios=False,
        remove_invalid_goals=False,
        log_names=None,
        log_labels=None,
        max_scenarios_per_log=None,
        scenario_types=scenario_types,
        scenario_tokens=None,
        map_name=None,
        limit_scenarios_per_type=limit_scenarios_per_type,
        subsample_ratio=subsample_ratio,
        limit_total_scenarios=None,
    )

    scenario_builder = NuPlanScenarioBuilder(DATA_VERSION, DATA_ROOT)
    scenarios = scenario_builder.get_scenarios(filter, RayDistributed())

    # Simulation Callbacks
    simulation_callback = SerializationCallback(
        output_directory=OUTPUT_DIR,
        folder_name=SIMULATION_DIR,
        serialize_into_single_file=True,
        serialization_type="msgpack",
    )

    data_dict = {"ads": {}}

    for scenario in tqdm(scenarios):
        name = scenario.scenario_name[:6]

        if name in names:
            continue
        names.append(name)

        if check_labels:
            ys = scene_labels[name]

        ## Metrics
        metric_engine = MetricsEngine(
            scenario_type=scenario.scenario_type,
            main_save_path=METRIC_PATH,
            timestamp=EXPERIMENT_TIME,
        )
        metric_engine.add_metric(example_metric)
        extra_callbacks = [
            MetricCallBack(
                metric_engine=metric_engine, scenario_name=scenario.scenario_name
            )
        ]
        scenario_callbacks = [simulation_callback] + extra_callbacks

        ## Simulation setup
        # Perception
        if reactive_agents:
            observations = IDMAgentsObservation(
                target_velocity=10,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.0,
                decel_max=2.0,
                scenario=scenario,
            )
        else:
            observations = BoxObservation(scenario=scenario)

        # Ego Controller
        ego_controller = (
            LogPlaybackController(scenario=scenario)
            if follow_expert
            else PerfectTrackingController(scenario=scenario)
        )

        simulation_setup = SimulationSetup(
            simulation_manager=StepSimulationManager(scenario=scenario),
            observations=observations,
            ego_controller=ego_controller,
            scenario=scenario,
        )

        planner = PredictorPlanner(
            subsample_ratio,
            scenario,
            verbose=True,
            following_expert=follow_expert,
            exp=exp,
        )

        simulation = Simulation(
            simulation_setup=simulation_setup,
            planner=planner,
            callbacks=scenario_callbacks,
            enable_progress_bar=False,
            simulation_history_buffer_duration=simulation_history_buffer_duration,
        )

        print("######", name, "######")
        ## Run simluation
        simulation.run()
        data_dict["ads"][name] = np.array(planner.advalues)

    np.save(exp_data_folder + exp_name + ".npy", data_dict)
