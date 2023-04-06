import jax.numpy as jnp
import numpy as np

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly

import hj_reachability as hj
from dynSimpleCarAvoid import DynSimpleCarCAvoid, CarPedAvoid


def value_function_vehicle():
    # make sure box is larger than set im interested in
    dynamics = DynSimpleCarCAvoid()
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=np.array([-10.0, -10.0, -np.pi, -2.0, -2.0]),
            hi=np.array([10.0, 10.0, np.pi, 20.0, 20.0]),
        ),
        (21, 21, 20, 23, 23),
        #    (20, 20, 5, 3, 3),
        periodic_dims=2,
    )

    values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 2
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
    )

    time = 0.0
    target_time = -2.0
    target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time)

    data = {"grid": grid, "target_values": target_values, "target_time": target_time}
    np.save("target_values/hj_reachability_values_veh.npy", data)


def value_function_ped():
    # make sure box is larger than set im interested in
    dynamics = CarPedAvoid()
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(
            lo=np.array([-10.0, -10.0, -np.pi, -2.0, -2.0]),
            hi=np.array([10.0, 10.0, np.pi, 20.0, 5.0]),
        ),
        (21, 21, 20, 23, 8),
        #    (5, 5, 5, 3, 3),
        periodic_dims=2,
    )

    values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 0.75
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
    )

    time = 0.0
    target_time = -2.0
    target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time)

    data = {"grid": grid, "target_values": target_values, "target_time": target_time}
    np.save("target_values/hj_reachability_values_ped.npy", data)


value_function_vehicle()
value_function_ped()
