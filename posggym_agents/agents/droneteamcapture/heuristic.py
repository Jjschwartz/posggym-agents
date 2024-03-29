"""Shortest path policy for PursuitEvasion env."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING, Tuple, List, Union

from enum import Enum

# from posggym.envs.gr
from posggym.envs.continuous.drone_team_capture import (
    DTCAction,
    DTCState,
    DTCModel,
)

import math
import numpy as np


from posggym_agents.policy import FullyObservablePolicy, PolicyID, PolicyState


if TYPE_CHECKING:
    from posggym.envs.continuous.core import Position
    from posggym.model import AgentID


class Algs(Enum):
    angelani2012 = 0
    janosov2017 = 1
    deviated_pure_pursuit = 4


class DroneTeamHeuristic(FullyObservablePolicy[DTCAction, DTCState]):
    """Heuristic Policies for the Pursuit Evasion environment.

    This policy sets the preferred action as the one which is on the shortest
    path to the evaders goal and which doesn't leave agent in same position.

    janosov2017 and angelani2012 are the non-holonomic adaptions of their original
    work following the protocol described in:
    C. de Souza, R. Newbury, A. Cosgun, P. Castillo, B. Vidolov and D. Kulić,
    "Decentralized Multi-Agent Pursuit Using Deep Reinforcement Learning,"
    in IEEE Robotics and Automation Letters, vol. 6, no. 3, pp. 4552-4559,
    July 2021, doi: 10.1109/LRA.2021.3068952.

    angelani2012 is the policy from this work:
    Angelani, L. (09 2012). Collective Predation and Escape Strategies.
    Physical Review Letters, 109. doi:10.1103/PhysRevLett.109.118104

    janosov2017 is the policy from this work:
    Janosov, M., Virágh, C., Vásárhelyi, G., & Vicsek, T. (2017). Group chasing tactics:
    how to catch a faster prey. New Journal of Physics, 19(5), 053003.
    doi:10.1088/1367-2630/aa69e7

    deviated_pure_pursuit is from this work:
    Souza, C., Castillo, P., & Vidolov, B. (2022). Local interaction and navigation
    guidance for hunters drones: a chase behavior approach with real-time tests.
    Robotica, 40(8), 2697–2715.
    """

    def __init__(
        self,
        model: DTCModel,
        agent_id: AgentID,
        policy_id: PolicyID,
        alg: Union[str, Algs] = "angelani2012",
    ):
        super().__init__(model, agent_id, policy_id)
        if type(alg) is str:
            try:
                self.alg = Algs[alg.lower()]
            except KeyError:
                raise AssertionError(
                    (
                        "Invalid Heuristic. Supported Options:",
                        f"{[str(x.name) for x in Algs]}",
                    )
                )
        elif type(alg) is Algs:
            self.alg = alg

        self._grid = self.model.grid

        self.omega_max = math.pi / 10
        self.cap_rad = 25
        self.vel_pur = 10
        self.vel_tar = 10

    def dpp_single(
        self,
        idx: int,
        all_pursuers: Tuple[Position, ...],
        target: Position,
        offset: float = math.pi / 8,
    ):
        sense = 0.0
        n_pursuers = len(all_pursuers)

        for j in range(n_pursuers):
            if j != idx:
                sense += self.delta(all_pursuers[idx], all_pursuers[j], target)

        alphaiT, _, _, _, _ = self.engagmment(all_pursuers[idx], target)
        omega_i = 0.6 * (alphaiT - sense * offset)

        omega_i = self.sat(omega_i, -self.omega_max, self.omega_max)
        return omega_i

    def DPP_group(
        self, Pursuer: Tuple[Position], target: Position, offset: float = math.pi / 8
    ) -> List[float]:
        omega = []
        n_pursuers = len(Pursuer)

        for i in range(n_pursuers):
            sense = 0.0
            for j in range(n_pursuers):
                if i != j:
                    sense += self.delta(Pursuer[i], Pursuer[j], target)

            alphaiT, _, _ = Pursuer[i].engagement2(target)
            omega_i = 0.6 * (alphaiT - sense * offset)

            omega_i = self.sat(omega_i, -self.omega_max, self.omega_max)

            omega.append(omega_i)
        return omega

    def euclidean_dist(self, coord1: Position, coord2: Position) -> float:
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

    def delta(self, Pi: Position, Pj: Position, target: Position) -> float:
        sense = 0
        _, Los_angle, _, _, _ = self.engagmment(Pi, target)

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

    def Rot(self, Los_angle: float) -> np.ndarray:
        R = np.array(
            [
                [math.cos(Los_angle), math.sin(Los_angle)],
                [-math.sin(Los_angle), math.cos(Los_angle)],
            ]
        )
        return R

    def sat(self, val: float, min: float, max: float) -> float:
        if val >= max:
            val = max
        if val <= min:
            val = min
        return val

    def PP(self, alpha: float) -> float:
        Kpp = 100
        omega = Kpp * alpha
        omega = self.sat(omega, -self.omega_max, self.omega_max)
        return omega

    def PNG(self, Angle_rate: float) -> float:
        N = 100
        omega = N * Angle_rate
        omega = self.sat(omega, -self.omega_max, self.omega_max)

        return omega

    def angelani2012_nh_single(
        self,
        Pursuer: Tuple[Position, ...],
        Pursuer_prev: Tuple[Position, ...],
        target: Position,
        idx: int,
    ) -> float:
        Vx, Vy = self.angelani2012_single(Pursuer, Pursuer_prev, target, idx)

        R = self.Rot(Pursuer[idx][2])
        Direction = R.dot([Vx, Vy])
        alpha = math.atan2(Direction[1], Direction[0])
        omega_i = self.PP(alpha)
        return omega_i

    def angelani2012_single(
        self,
        Pursuer: Tuple[Position, ...],
        Pursuer_prev: Tuple[Position, ...],
        target: Position,
        idx: int,
    ) -> Tuple[float, float]:
        Rep = self.repulsion(Pursuer, idx)
        Align = self.alignment(Pursuer, Pursuer_prev)
        Atrac = self.attraction(Pursuer[idx], target)

        Vx_i = Rep[0] + Align[0] + Atrac[0]
        Vy_i = Rep[1] + Align[1] + Atrac[1]

        return Vx_i, Vy_i

    def repulsion(self, Pursuer: Tuple[Position, ...], i: int) -> List[float]:
        r = 300
        Dx = 0.0
        Dy = 0.0

        for j in range(len(Pursuer)):
            if j != i:
                r_iT = np.array(Pursuer[j])[:2] - np.array(Pursuer[i])[:2]
                dist = np.linalg.norm(r_iT)
                if dist < r:
                    dx, dy = self.rep_force(r_iT, dist)
                    Dx += dx
                    Dy += dy

        dx, dy = self.normalise(Dx, Dy)

        return [dx, dy]

    def rep_force(self, r_ij: np.ndarray, dist: float) -> Tuple[float, float]:
        sigma = 3
        u = -r_ij / dist
        den = 1 + math.exp((dist - 20) / sigma)

        rep = u / den

        return rep[0], rep[1]

    def alignment(
        self, Pursuer: Tuple[Position, ...], Pursuer_prev: Tuple[Position, ...]
    ) -> List[float]:
        dx = 0.0
        dy = 0.0

        for i in range(len(Pursuer)):
            dx = dx + (Pursuer[i][0] - Pursuer_prev[i][0])
            dy = dy + (Pursuer[i][1] - Pursuer_prev[i][1])

        dx, dy = self.normalise(dx, dy)

        return [dx, dy]

    def attraction(self, Pursuer_i: Position, target: Position) -> List[float]:
        r_iT = np.array(target[:2]) - np.array(Pursuer_i[:2])

        dx, dy = self.normalise(r_iT[0], r_iT[1])

        return [dx, dy]

    def normalise(self, dx: float, dy: float) -> Tuple[float, float]:
        d = math.sqrt(dx**2 + dy**2)
        d = self.sat(d, 0.000001, 10000000)
        return dx / d, dy / d

    def janosov2017_single(
        self,
        pursuer_coord: Tuple[Position, ...],
        pursuer_prev_coords: Tuple[Position, ...],
        target: Position,
        target_prev: Position,
        idx: int,
    ) -> Tuple[float, float]:
        arena = self.arena()
        coll = [0, 0]

        chase = self.attraction2(
            pursuer_coord[idx], pursuer_prev_coords[idx], target, target_prev
        )
        inter = self.alignment2(pursuer_coord, pursuer_prev_coords, idx)

        Vx_i = arena[0] + coll[0] + chase[0] + inter[0]
        Vy_i = arena[1] + coll[1] + chase[1] + inter[1]

        return Vx_i, Vy_i

    def janosov2017_single_nh(
        self,
        pursuer_coord: Tuple[Position, ...],
        pursuer_prev_coords: Tuple[Position, ...],
        target: Position,
        target_prev: Position,
        idx: int,
    ) -> float:
        vx, vy = self.janosov2017_single(
            pursuer_coord, pursuer_prev_coords, target, target_prev, idx
        )

        R = self.Rot(pursuer_coord[idx][2])
        Direction = R.dot([vx, vy])
        alpha = math.atan2(Direction[1], Direction[0])
        omega = self.PP(alpha)
        return omega

    def arena(self) -> List[float]:
        return [0.0, 0.0]

    def attraction2(
        self,
        Pursuer_i: Position,
        Pursuer_i_prev: Position,
        target: Position,
        target_prev: Position,
    ):
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

    def prediction(self, Pursuer_i: Position, target: Position, vel_t: List[float]):
        pos_pred = target

        dist = self.euclidean_dist(pos_pred, Pursuer_i)

        tau = 20
        time_pred = dist / self.vel_tar
        time_pred = self.sat(time_pred, 0, tau)

        vel_t = self.normalise(vel_t[0], vel_t[1])  # type: ignore
        vel_x = vel_t[0] * self.vel_tar
        vel_y = vel_t[1] * self.vel_tar

        pos_pred_x = target[0] + vel_x * time_pred
        pos_pred_y = target[1] + vel_y * time_pred

        return [pos_pred_x, pos_pred_y]

    def alignment2(self, Pursuer, Pursuer_prev, i):
        inte_x = 0
        inte_y = 0
        rad_inter = 250
        C_inter = 0.5
        C_f = 0.5

        for j in range(len(Pursuer)):
            if j != i:
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

    def step(self, state: DTCState) -> DTCAction:
        idx = int(self.agent_id)
        angle = 0.0
        if self.alg == Algs.janosov2017:
            angle = self.janosov2017_single_nh(
                state.pursuer_coords,
                state.prev_pursuer_coords,
                state.target_coords,
                state.prev_target_coords,
                idx,
            )
        elif self.alg == Algs.angelani2012:
            angle = self.angelani2012_nh_single(
                state.pursuer_coords,
                state.prev_pursuer_coords,
                state.target_coords,
                idx,
            )
        elif self.alg == Algs.deviated_pure_pursuit:
            angle = self.dpp_single(idx, state.pursuer_coords, state.target_coords)
        else:
            return NotImplementedError("{self.alg} is not implemented")

        return np.array([angle], dtype=np.float32)

    def get_initial_state(self) -> PolicyState:
        state = super().get_initial_state()
        state["last_state"] = None
        return state

    def get_next_state(
        self,
        obs: DTCState,
        state: PolicyState,
    ) -> PolicyState:
        return {"last_state": obs}

    def sample_action(self, state: PolicyState) -> DTCAction:
        return self.step(state["last_state"])

    def get_pi(self, state: PolicyState) -> Dict[DTCAction, float]:
        raise NotImplementedError(
            f"`get_pi()` not implemented by {self.__class__.__name__} policy"
        )

    def get_value(self, state: PolicyState) -> float:
        raise NotImplementedError(
            f"`get_value()` not implemented by {self.__class__.__name__} policy"
        )
