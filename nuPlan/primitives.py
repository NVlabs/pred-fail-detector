import sys

sys.path.append("../")

import numpy as np
from detection.query import Query
from detection.query_utils import query_node
from copy import deepcopy


def propagate_state(vh, h, dh, a, dt):
    xtp1 = (vh * dt + 0.5 * a * (dt**2)) * np.cos(dh * dt)
    ytp1 = (vh * dt + 0.5 * a * (dt**2)) * np.sin(dh * dt)
    vhtp1 = vh + a * dt
    htp1 = h + dh * dt
    return xtp1, ytp1, vhtp1, htp1


class Primitive_Generator:
    def __init__(self, dt=0.5, ph=4, scenario=None):
        self.dt = dt
        self.ph = ph

        self.acceleration_options = np.array([-2, 0, 1])
        self.heading_rate_options = np.array([-0.3, -0.1, 0, 0.1, 0.3]) * 1

        # [acceleration_control, heading_rate control]
        self.control_sequence = self._get_control_action_sequences()

    def _get_control_action_sequences(self):
        ao = self.acceleration_options
        hro = self.heading_rate_options
        c = [[0, 0]]
        a1 = np.tile(ao, len(hro))
        hr1 = np.repeat(hro, len(ao))
        c1 = np.expand_dims(np.stack((a1, hr1)).T, axis=1)

        for t in range(self.ph):
            if t == 0:
                pass
            elif t == 1:
                c = np.tile(c, (len(c1), 1, 1))
                c = np.concatenate((c, c1), axis=1)
            elif t > 1:
                c21 = np.tile(c, (len(c1), 1, 1))
                c22 = np.repeat(c1, len(c), axis=0)
                c = np.concatenate((c21, c22), axis=1)

        return c

    def generate_motion_primitives(self, ego_node, timestep, return_control=False):
        queries = [
            Query.lon_velocity,
            Query.lon_acceleration,
            Query.heading,
            Query.heading_rate,
        ]
        states = query_node(ego_node, queries, timestep)
        c = deepcopy(self.control_sequence)

        vht = states["lon_velocity"]
        ht = states["heading"]
        at = states["lon_acceleration"]
        dht = states["heading_rate"]

        c[:, 0, :] += [at[0], dht[0]]

        primitives = np.zeros((c.shape[0], c.shape[1], 3))
        for i in range(self.ph):
            at = c[:, i, 0]
            dht = c[:, i, 1]
            xtp1, ytp1, vhtp1, htp1 = propagate_state(vht, ht, dht, at, self.dt)
            temp = np.stack((xtp1, ytp1, htp1 - ht), axis=-1)
            primitives[:, i, :] = temp
            vht, ht = vhtp1, htp1

        if return_control:
            return primitives, c

        return primitives
