import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets


class DynSimpleCarCAvoid(dynamics.ControlAndDisturbanceAffineDynamics):
    """Relative dynamics of a two-agent system where each is modelled as a dynamically extended simple car model.
    Dynamically extended simple car model:

    dx   = v * cos(phi)
    dy   = v * sin(phi)
    dphi = v * tan(d) / L
    dv   = a

    (Let p = tan(d) and this will become control affine)

    Relative (R) dynamics:
        zRdot = [-vA + vB * cos(phiR) + yR * vA tan(dA) / LA,
                  vB * sin(phiR) - xR * vA tan(dA) / LA,
                  vB tan(dB) / LB - vA tan(dA) / LA,
                  aA,
                  aB],

        where:
            xR = (xB - xA) * cos(phiA)  + (yB - yA) * sin(phiA)
            yR = -(xB - xA) * sin(phiA)  + (yB - yA) * cos(phiA)
            phiR = phiB - phiA
            vA
            vB

    Control of agent (A):
        uA = [tan(dA), aA]

        where:
            dA in [-d_maxA, d_maxA]
            aA in [aminA, amaxA]

    Control of agent (B):
        uB = [tan(dB), aB]

        where:
            dB in [-d_maxB, d_maxB]
            aB in [aminB, amaxB]
    """

    def __init__(
        self,
        evader_length=1,
        pursuer_length=1,
        evader_accel_bounds=[-2.0, 1.0],
        pursuer_accel_bounds=[-2.0, 1.0],
        evader_max_steering=jnp.tan(10 * jnp.pi / 180),
        pursuer_max_steering=jnp.tan(10 * jnp.pi / 180),
        control_mode="max",
        disturbance_mode="min",
        control_space=None,
        disturbance_space=None,
    ):

        self.evader_length = evader_length
        self.pursuer_length = pursuer_length

        if control_space is None:
            control_space = sets.Box(
                lo=jnp.array([-evader_max_steering, evader_accel_bounds[0]]),
                hi=jnp.array([evader_max_steering, evader_accel_bounds[1]]),
            )

        if disturbance_space is None:
            disturbance_space = sets.Box(
                lo=jnp.array([-pursuer_max_steering, pursuer_accel_bounds[0]]),
                hi=jnp.array([pursuer_max_steering, pursuer_accel_bounds[1]]),
            )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        _, _, psiR, vA, vB = state
        return jnp.array([-vA + vB * jnp.cos(psiR), vB * jnp.sin(psiR), 0.0, 0.0, 0.0])

    def control_jacobian(self, state, time):
        """
        uA = [tan(dA), aA]
        """
        xR, yR, _, vA, _ = state
        return jnp.array(
            [
                [vA * yR / self.evader_length, 0.0],
                [-vA * xR / self.evader_length, 0.0],
                [-vA / self.evader_length, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

    def disturbance_jacobian(self, state, time):
        _, _, _, _, vB = state
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [vB / self.pursuer_length, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )

    def get_global_disturbance_state(self, stateA, stateR):
        """Given the current state of agent A (control) and the relative state,
        returns agent B (disturbance)'s state in global coordinates.
        """
        _, _, psiA, _ = stateA
        _, _, _, _, vB = stateR

        rot_mat = jnp.array(
            [
                [jnp.cos(psiA), jnp.sin(psiA), 0.0],
                [-jnp.sin(psiA), jnp.cos(psiA), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        inv_rot_mat = jnp.linalg.inv(rot_mat)

        # res = [xB - xA; yB - yA; psiB - psiA]
        res = jnp.matmul(inv_rot_mat, stateR[0:3])

        # extract just the parts of res that correspond to agent B's state
        # and form stateB = [xB, yB, psiB, vB]
        stateB = jnp.append(res + stateA[0:3], vB)

        return stateB

    def get_rel_state(self, stateA, stateB):
        """Given the current state of agent A (control) and
        agent B (disturbance), computes the relative state vector.
        """
        _, _, psiA, vA = stateA
        _, _, psiB, vB = stateB

        rot_mat = jnp.array(
            [[jnp.cos(psiA), jnp.sin(psiA)], [-jnp.sin(psiA), jnp.cos(psiA)]]
        )
        posR = jnp.matmul(rot_mat, stateB[0:2] - stateA[0:2])

        return jnp.append(posR, jnp.array([psiB - psiA, vA, vB]))


class CarPedAvoid(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(
        self,
        evader_length=1,
        pursuer_length=0.2,
        evader_accel_bounds=[-2.0, 1.0],
        pursuer_accel_bounds=[-0.5, 0.5],
        evader_max_steering=jnp.tan(10 * jnp.pi / 180),
        pursuer_max_steering=jnp.tan(180 * jnp.pi / 180),
        control_mode="max",
        disturbance_mode="min",
        control_space=None,
        disturbance_space=None,
    ):

        self.evader_length = evader_length
        self.pursuer_length = pursuer_length

        if control_space is None:
            control_space = sets.Box(
                lo=jnp.array([-evader_max_steering, evader_accel_bounds[0]]),
                hi=jnp.array([evader_max_steering, evader_accel_bounds[1]]),
            )

        if disturbance_space is None:
            disturbance_space = sets.Box(
                lo=jnp.array([-pursuer_max_steering, pursuer_accel_bounds[0]]),
                hi=jnp.array([pursuer_max_steering, pursuer_accel_bounds[1]]),
            )

        super().__init__(
            control_mode, disturbance_mode, control_space, disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        _, _, psiR, vA, vB = state
        return jnp.array([-vA + vB * jnp.cos(psiR), vB * jnp.sin(psiR), 0.0, 0.0, 0.0])

    def control_jacobian(self, state, time):
        """
        uA = [tan(dA), aA]
        """
        xR, yR, _, vA, _ = state
        return jnp.array(
            [
                [vA * yR / self.evader_length, 0.0],
                [-vA * xR / self.evader_length, 0.0],
                [-vA / self.evader_length, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )

    def disturbance_jacobian(self, state, time):
        _, _, _, _, vB = state
        return jnp.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [vB / self.pursuer_length, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )

    def get_global_disturbance_state(self, stateA, stateR):
        """Given the current state of agent A (control) and the relative state,
        returns agent B (disturbance)'s state in global coordinates.
        """
        _, _, psiA, _ = stateA
        _, _, _, _, vB = stateR

        rot_mat = jnp.array(
            [
                [jnp.cos(psiA), jnp.sin(psiA), 0.0],
                [-jnp.sin(psiA), jnp.cos(psiA), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        inv_rot_mat = jnp.linalg.inv(rot_mat)

        # res = [xB - xA; yB - yA; psiB - psiA]
        res = jnp.matmul(inv_rot_mat, stateR[0:3])

        # extract just the parts of res that correspond to agent B's state
        # and form stateB = [xB, yB, psiB, vB]
        stateB = jnp.append(res + stateA[0:3], vB)

        return stateB

    def get_rel_state(self, stateA, stateB):
        """Given the current state of agent A (control) and
        agent B (disturbance), computes the relative state vector.
        """
        _, _, psiA, vA = stateA
        _, _, psiB, vB = stateB

        rot_mat = jnp.array(
            [[jnp.cos(psiA), jnp.sin(psiA)], [-jnp.sin(psiA), jnp.cos(psiA)]]
        )
        posR = jnp.matmul(rot_mat, stateB[0:2] - stateA[0:2])

        return jnp.append(posR, jnp.array([psiB - psiA, vA, vB]))
