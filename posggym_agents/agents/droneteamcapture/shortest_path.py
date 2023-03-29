"""Shortest path policy for PursuitEvasion env."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING, Tuple

# from posggym.envs.gr
from posggym.envs.continuous.drone_team_capture import (
    DTCObs,
    DTCAction,
    DTCState,
    DTCModel,
)

import math
import numpy as np


from posggym_agents.policy import Policy, PolicyID, PolicyState


if TYPE_CHECKING:
    from posggym.envs.continuous.core import Position
    from posggym.model import AgentID
    from posggym.utils.history import AgentHistory


class DroneTeamHeuristic(Policy[DTCAction, DTCObs]):
    """Shortest Path Policy for the Pursuit Evasion environment.

    This policy sets the preferred action as the one which is on the shortest
    path to the evaders goal and which doesn't leave agent in same position.

    """

    def __init__(self, model: DTCModel, agent_id: AgentID, policy_id: PolicyID):
        super().__init__(model, agent_id, policy_id)

        # assert all(p == 1.0 for p in self.model._action_probs), (
        #     f"{self.__class__.__name__} only supported for deterministic versions of "
        #     "the PursuitEvasion environment."
        # )

        self._grid = self.model.grid
        # self._is_evader = self.agent_id == self.model.EVADER_IDX
        # self._action_space = list(range(self.model.action_spaces[self.agent_id].n))

        # evader_end_coords = list(
        #     set(self._grid.evader_start_coords + self._grid.all_goal_coords)
        # )
        # self._dists = self._grid.get_all_shortest_paths(evader_end_coords)
        self.omega_max = math.pi / 10
        self.cap_rad = 25
        self.vel_pur = 10
        self.vel_tar = 10

    def dpp_single(
        self,
        idx: int,
        all_pursuers: Tuple[Position, ...],
        target: Position,
        n_pursuers: int,
        offset: float = math.pi / 8,
    ):
        sense = 0

        for j in range(n_pursuers):
            if j != idx:
                sense += self.delta(all_pursuers[idx], all_pursuers[j], target)

        alphaiT, ang_r_t, dist, _, _ = self.engagmment(all_pursuers[idx], target)
        print(alphaiT)
        omega_i = 0.6 * (alphaiT - sense * offset)

        omega_i = self.sat(omega_i, -self.omega_max, self.omega_max)
        return omega_i

    def DPP_group(self, Pursuer, target, offset=0):
        omega = []
        n_pursuers = len(Pursuer)

        for i in range(n_pursuers):
            sense = 0
            for j in range(n_pursuers):
                if i != j:
                    sense += self.delta(Pursuer[i], Pursuer[j], target)

            alphaiT, ang_r_t, dist = Pursuer[i].engagement2(target)
            omega_i = 0.6 * (alphaiT - sense * offset)

            omega_i = self.sat(omega_i, -self.omega_max, self.omega_max)

            omega.append(omega_i)
        return omega

    def euclidean_dist(self, coord1: Position, coord2: Position):
        return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

    def engagmment(
        self, agent_i: Position, agent_j: Position, dist_factor: float = 1.0
    ) -> Tuple[float, ...]:
        dist = self.euclidean_dist(agent_i, agent_j)

        yaw = agent_i[2]

        # Rotation matrix of yaw
        R = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])

        T_p = np.array([agent_j[0] - agent_i[0], agent_j[1] - agent_i[1]])
        Los_angle = math.atan2(T_p[1], T_p[0])

        T_p = R.dot(T_p)
        alpha = math.atan2(T_p[1], T_p[0])

        return (
            alpha,
            Los_angle,
            dist / dist_factor,
            T_p[0] / dist_factor,
            T_p[1] / dist_factor,
        )

    def delta(self, Pi, Pj, target):
        sense = 0
        alpha, Los_angle, dist, T_p, T_p1 = self.engagmment(Pi, target)
        # Pi.engagement2(target)

        R = np.array(
            [
                [math.cos(Los_angle), math.sin(Los_angle)],
                [-math.sin(Los_angle), math.cos(Los_angle)],
            ]
        )

        i_j = np.array(Pj[:2]) - np.array(Pi[:2])
        i_j = R.dot(i_j)

        if abs(i_j[1]) != 0:
            sense = i_j[1] / abs(i_j[1])  # for do not divide by zero

        return sense

    def Rot(self, Los_angle):
        R = np.array(
            [
                [math.cos(Los_angle), math.sin(Los_angle)],
                [-math.sin(Los_angle), math.cos(Los_angle)],
            ]
        )
        return R

    def sat(self, val, min, max):
        if val >= max:
            val = max
        if val <= min:
            val = min
        return val

    def PP(self, alpha):
        Kpp = 100
        omega = Kpp * alpha
        omega = self.sat(omega, -self.omega_max, self.omega_max)
        return omega

    def PNG(self, Angle_rate):
        N = 100
        omega = N * Angle_rate
        omega = self.sat(omega, -self.omega_max, self.omega_max)

        return omega

    def Mixte_pursuit(self, Pursuer, target):
        omega = []
        n_pursuers = len(Pursuer)
        offset = math.pi / 12
        # if n_pursuers ==1: offset=0
        # else: offset = math.pi / (3*n_pursuers)

        r = 250  # to perceive others
        dist_min = r
        for i in range(n_pursuers):
            sense = 0
            for j in range(n_pursuers):
                if i != j:
                    sense += self.delta(Pursuer[i], Pursuer[j], target)
                    dist_ij = np.linalg.norm(Pursuer[i].pos() - Pursuer[j].pos())
                    if dist_ij < dist_min:
                        dist_min = dist_ij

            dist_iT = np.linalg.norm(Pursuer[i].pos() - target.pos())
            r_min = self.sat(dist_iT, 0, r)
            K = (r_min - dist_min) / r_min
            K = self.sat(K, 0.1, 10)

            alphaiT, ang_r_t, dist = Pursuer[i].engagement(target)
            err_alpha = alphaiT - sense * offset
            err_alpha = self.sat(err_alpha, -math.pi / 2, math.pi / 2)
            DPP = K**2 * math.tan(err_alpha)

            PN = 10 * ang_r_t

            # print(PN, DPP)
            PN = self.sat(PN, -0.4, 0.4)
            DPP = self.sat(DPP, -1, 1)

            omega_i = PN + DPP
            omega_i = self.sat(omega_i, -self.omega_max, self.omega_max)
            omega.append(omega_i)

        return omega

    def Predators_1_n(self, Pursuer, Pursuer_prev, target):
        omega = []
        n_pursuers = len(Pursuer)
        Vx, Vy = self.Predators_1(Pursuer, Pursuer_prev, target)

        for i in range(n_pursuers):
            R = self.Rot(Pursuer[i].get_yaw()[0])
            Direction = R.dot([Vx[i], Vy[i]])
            alpha = math.atan2(Direction[1], Direction[0])
            omega_i = self.PP(alpha)
            omega.append(omega_i)
        return omega

    def Predators_1(self, Pursuer, Pursuer_prev, target):
        Vx = []
        Vy = []
        n_pursuers = len(Pursuer)

        for i in range(n_pursuers):
            Rep = self.repulsion(Pursuer, i)
            Align = self.alignment(Pursuer, Pursuer_prev)
            Atrac = self.attraction(Pursuer[i].pos(), target.pos())

            Vx_i = Rep[0] + Align[0] + Atrac[0]
            Vy_i = Rep[1] + Align[1] + Atrac[1]

            # omega_i = self.vel2rate(Pursuer[i], [Vx,Vy])
            Vx.append(Vx_i)
            Vy.append(Vy_i)
        return Vx, Vy

    def repulsion(self, Pursuer, i):
        r = 300
        Dx = 0
        Dy = 0

        for j in range(len(Pursuer)):
            if j != i:
                r_iT = Pursuer[j].pos() - Pursuer[i].pos()
                dist = np.linalg.norm(r_iT)
                if dist < r:
                    dx, dy = self.rep_force(r_iT, dist)
                    Dx += dx
                    Dy += dy

        dx, dy = self.normalise(Dx, Dy)

        return [dx, dy]

    def rep_force(self, r_ij, dist):
        sigma = 3
        u = -r_ij / dist
        den = 1 + math.exp((dist - 20) / sigma)

        rep = u / den

        return rep[0], rep[1]

    def alignment(self, Pursuer, Pursuer_prev):
        dx = 0
        dy = 0

        for i in range(len(Pursuer)):
            dx = dx + (Pursuer[i].pos()[0] - Pursuer_prev[i].pos()[0])
            dy = dy + (Pursuer[i].pos()[1] - Pursuer_prev[i].pos()[1])

        dx, dy = self.normalise(dx, dy)

        return [dx, dy]

    def attraction(self, Pursuer_i, target):
        r_iT = np.array(target[:2]) - np.array(Pursuer_i[:2])

        dx, dy = self.normalise(r_iT[0], r_iT[1])

        return [dx, dy]

    def normalise(self, dx, dy):
        d = math.sqrt(dx**2 + dy**2)
        d = self.sat(d, 0.000001, 10000000)
        return dx / d, dy / d

    def Predators_2_n(self, Pursuer, Pursuer_prev, target, target_prev):
        omega = []
        n_pursuers = len(Pursuer)
        Vx, Vy = self.Predators_2(Pursuer, Pursuer_prev, target, target_prev)

        for i in range(n_pursuers):
            R = self.Rot(Pursuer[i].get_yaw()[0])
            Direction = R.dot([Vx[i], Vy[i]])
            alpha = math.atan2(Direction[1], Direction[0])
            omega_i = self.PP(alpha)
            omega.append(omega_i)
        return omega

    def Predators_2_single(
        self,
        pursuer_coord: Tuple[Position, ...],
        pursuer_prev_coords: Tuple[Position, ...],
        target: Position,
        target_prev: Position,
        idx: int,
    ) -> Tuple[float, float]:
        arena = self.arena(pursuer_coord, idx)
        coll = [0, 0]

        chase = self.attraction2(
            pursuer_coord[idx], pursuer_prev_coords[idx], target, target_prev
        )
        inter = self.alignment2(pursuer_coord, pursuer_prev_coords, idx)

        Vx_i = arena[0] + coll[0] + chase[0] + inter[0]
        Vy_i = arena[1] + coll[1] + chase[1] + inter[1]

        return Vx_i, Vy_i

    def Predators_2_single_nh(
        self,
        pursuer_coord: Tuple[Position, ...],
        pursuer_prev_coords: Tuple[Position, ...],
        target: Position,
        target_prev: Position,
        idx: int,
    ):
        vx, vy = self.Predators_2_single(
            pursuer_coord, pursuer_prev_coords, target, target_prev, idx
        )

        R = self.Rot(pursuer_coord[idx][2])
        Direction = R.dot([vx, vy])
        alpha = math.atan2(Direction[1], Direction[0])
        print(alpha)
        omega = self.PP(alpha)
        return omega

    def Predators_2(self, Pursuer, Pursuer_prev, target, target_prev):
        Vx = []
        Vy = []
        n_pursuers = len(Pursuer)

        for i in range(n_pursuers):
            arena = self.arena(Pursuer, i)
            coll = [0, 0]  # self.repulsion2(Pursuer, i)
            chase = self.attraction2(Pursuer[i], Pursuer_prev[i], target, target_prev)
            inter = self.alignment2(Pursuer, Pursuer_prev, i)

            Vx_i = arena[0] + coll[0] + chase[0] + inter[0]
            Vy_i = arena[1] + coll[1] + chase[1] + inter[1]

            Vx.append(Vx_i)
            Vy.append(Vy_i)

        return Vx, Vy

    def arena(self, Pursuer, i):
        return [0, 0]

    def repulsion2(self, Pursuer, i):
        rep = self.repulsion(Pursuer, i)
        return [rep[0], rep[1]]

    def attraction2(self, Pursuer_i, Pursuer_i_prev, target, target_prev):
        # Friction term
        dist = self.euclidean_dist(target, Pursuer_i)
        vel_p = (np.array(Pursuer_i) - np.array(Pursuer_i_prev))[:2]
        vel_t = (np.array(target) - np.array(target_prev))[:2]
        visc = (vel_p - vel_t) / dist**2

        # Atraction term
        target_pred = self.prediction(Pursuer_i, target, vel_t)
        atrac = self.attraction(Pursuer_i, target_pred)

        chase = atrac + 1.5 * visc
        chase = self.normalise(chase[0], chase[1])

        chase_x = chase[0] * self.vel_pur
        chase_y = chase[1] * self.vel_pur

        return [chase_x, chase_y]

    def prediction(self, Pursuer_i, target, vel_t):
        pos_pred = target

        dist = self.euclidean_dist(pos_pred, Pursuer_i)

        tau = 20
        time_pred = dist / self.vel_tar
        time_pred = self.sat(time_pred, 0, tau)

        vel_t = self.normalise(vel_t[0], vel_t[1])
        vel_x = vel_t[0] * self.vel_tar
        vel_y = vel_t[1] * self.vel_tar

        pos_pred_x = target[0] + vel_x * time_pred
        pos_pred_y = target[1] + vel_y * time_pred

        # print('Pos, pred, vel: ', target.pos()[0], pos_pred_x, vel_x)

        return [pos_pred_x, pos_pred_y]

    def alignment2(self, Pursuer, Pursuer_prev, i):
        inte_x = 0
        inte_y = 0
        rad_inter = 250
        C_inter = 0.5
        C_f = 0.5

        for j in range(len(Pursuer)):
            if j != i:
                # dx = dx + (Pursuer[i].pos()[0] - Pursuer_prev[i].pos()[0])
                # dy = dy + (Pursuer[i].pos()[1] - Pursuer_prev[i].pos()[1])
                d_ij = (np.array(Pursuer[i]) - np.array(Pursuer[j]))[:2]
                d = np.linalg.norm(d_ij)

                d_ij = self.normalise(d_ij[0], d_ij[1])

                vel_i = np.array(Pursuer[i]) - np.array(Pursuer_prev[i])
                vel_j = np.array(Pursuer[j]) - np.array(Pursuer_prev[j])

                rep_x = d_ij[0] * self.sat((d - rad_inter), -10000, 0) / d
                rep_y = d_ij[1] * self.sat((d - rad_inter), -10000, 0) / d

                fric_x = (vel_i[0] - vel_j[0]) / d**2
                fric_y = (vel_i[1] - vel_j[1]) / d**2

                inte_x += -rep_x + C_f * fric_x
                inte_y += -rep_y + C_f * fric_y

        inte_x, inte_y = self.normalise(inte_x, inte_y)

        dx = C_inter * inte_x * self.vel_pur
        dy = C_inter * inte_y * self.vel_pur

        return [dx, dy]

    def _step(self, state: DTCState | None, idx: int) -> DTCAction:
        return self.Predators_2_single_nh(
            state.pursuer_coords,
            state.prev_pursuer_coords,
            state.target_coords,
            state.prev_target_coords,
            idx,
        )
        # print(state)

    def step(self, obs: DTCObs | None) -> DTCAction:
        raise AssertionError("This requires more then just the regular obs")

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        return state

    def get_next_state(
        self,
        obs: DTCObs,
        state: PolicyState,
    ) -> PolicyState:
        return {}

    def get_state_from_history(self, history: AgentHistory) -> PolicyState:
        return {}

    def sample_action(self, state: PolicyState) -> DTCAction:
        return state["action"]

    def get_pi(self, state: PolicyState) -> Dict[DTCAction, float]:
        return {}

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` no implemented by {self.__class__.__name__} policy"
        )
