from typing import Optional, List, Tuple

import posggym.model as M
from posggym.utils.history import AgentHistory
from posggym.envs.grid_world.core import Coord
from posggym.envs.grid_world.pursuit_evasion.model import (
    PursuitEvasionModel, PEAction
)

import posggym_agents.policy as Pi


class PESPPolicy(Pi.BaseHiddenStatePolicy):
    """Shortest Path Policy for the Pursuit Evasion environment.

    This policy sets the preferred action as the one which is on the shortest
    path to the evaders goal and which doesnt leave agent in same position.
    """

    def __init__(self,
                 model: PursuitEvasionModel,
                 agent_id: M.AgentID,
                 policy_id: str):
        super().__init__(model, agent_id, policy_id)
        assert all(p == 1.0 for p in self.model._action_probs), (
            "PESPPolicy only supported for deterministic environment"
        )

        self._grid = self.model.grid
        self._is_evader = self.agent_id == self.model.EVADER_IDX
        self._action_space = list(
            range(self.model.action_spaces[self.agent_id].n)
        )

        self._update_num = 0
        # these will be set during first update
        self._coord = (0, 0)
        self._prev_coord = (0, 0)
        # goal coord if agent is evader else evader start coord
        self._target_coord = (0, 0)

        evader_end_coords = list(
            set(self._grid.evader_start_coords + self._grid.all_goal_coords)
        )
        self._dists = self._grid.get_all_shortest_paths(evader_end_coords)

    def get_action(self) -> M.Action:
        _, obs = self.history.get_last_step()
        return self._get_sp_action(
            self._coord, self._prev_coord, self._target_coord
        )

    def get_action_by_history(self, history: AgentHistory) -> M.Action:
        _, obs = history.get_last_step()
        coord, prev_coord = self._get_coord_from_history(history)
        # goal coord if agent is evader else evader start coord
        target_coord = obs[3] if self._is_evader else obs[1]
        return self._get_sp_action(coord, prev_coord, target_coord)

    def get_action_by_hidden_state(self,
                                   hidden_state: Pi.PolicyHiddenState
                                   ) -> M.Action:
        return self._get_sp_action(
            hidden_state["coord"],
            hidden_state["prev_coord"],
            hidden_state["target_coord"]
        )

    def get_pi(self,
               history: Optional[AgentHistory] = None
               ) -> Pi.ActionDist:
        coords = self._get_coord_and_target_coord(history)
        return self._get_pi_from_coords(*coords)

    def get_pi_from_hidden_state(self,
                                 hidden_state: Pi.PolicyHiddenState
                                 ) -> Pi.ActionDist:
        return self._get_pi_from_coords(
            hidden_state["coord"],
            hidden_state["prev_coord"],
            hidden_state["target_coord"]
        )

    def _get_pi_from_coords(self,
                            coord: Coord,
                            prev_coord: Coord,
                            goal_coord: Coord) -> Pi.ActionDist:
        dists = self._get_action_sp_dists(coord, prev_coord, goal_coord)

        pi = {a: 1.0 for a in self._action_space}
        max_dist, min_dist = max(dists), min(dists)
        dist_gap = max(1, max_dist - min_dist)

        weight_sum = 0.0
        for i, a in enumerate(self._action_space):
            weight = 1 - ((dists[i] - min_dist) / dist_gap)
            pi[a] = weight
            weight_sum += weight

        for a in self._action_space:
            pi[a] /= weight_sum

        return pi

    def get_next_hidden_state(self,
                              hidden_state: Pi.PolicyHiddenState,
                              action: M.Action,
                              obs: M.Observation
                              ) -> Pi.PolicyHiddenState:
        next_hidden_state = super().get_next_hidden_state(
            hidden_state, action, obs
        )
        if hidden_state["update_num"] == 0:
            next_hidden_state.update({
                "update_num": 1,
                "coord": obs[1] if self._is_evader else obs[2],
                "prev_coord": (0, 0),
                "target_coord": obs[3] if self._is_evader else obs[1]
            })
        else:
            next_hidden_state.update({
                "update_num": hidden_state["update_num"] + 1,
                "coord": self._grid.get_next_coord(
                    hidden_state["coord"], action, False
                ),
                "prev_coord": hidden_state["coord"],
                "target_coord": hidden_state["target_coord"]
            })
        return next_hidden_state

    def get_initial_hidden_state(self) -> Pi.PolicyHiddenState:
        hidden_state = super().get_initial_hidden_state()
        hidden_state.update({
            "update_num": 0,
            "coord": (0, 0),
            "prev_coord": (0, 0),
            "target_coord": (0, 0)
        })
        return hidden_state

    def get_hidden_state(self) -> Pi.PolicyHiddenState:
        hidden_state = super().get_hidden_state()
        hidden_state.update({
            "update_num": self._update_num,
            "coord": self._coord,
            "prev_coord": self._prev_coord,
            "target_coord": self._target_coord
        })
        return hidden_state

    def set_hidden_state(self, hidden_state: Pi.PolicyHiddenState):
        super().set_hidden_state(hidden_state)
        self._update_num = hidden_state["update_num"]
        self._coord = hidden_state["coord"]
        self._prev_coord = hidden_state["prev_coord"]
        self._target_coord = hidden_state["target_coord"]

    def update(self, action: M.Action, obs: M.Observation) -> None:
        super().update(action, obs)
        if self._update_num == 0:
            self._coord = obs[1] if self._is_evader else obs[2]
            self._target_coord = obs[3] if self._is_evader else obs[1]
        else:
            self._prev_coord = self._coord
            self._coord = self._grid.get_next_coord(self._coord, action, False)
        self._update_num += 1

    def reset(self) -> None:
        super().reset()
        self._update_num = 0
        self._coord = (0, 0)
        self._prev_coord = (0, 0)
        self._target_coord = (0, 0)

    def _get_sp_action(self,
                       coord: Coord,
                       prev_coord: Coord,
                       goal_coord: Coord) -> PEAction:
        sp_action = self._action_space[0]
        sp_dist = float('inf')
        for a in self._action_space:
            next_coord = self._grid.get_next_coord(coord, a, False)
            if next_coord == prev_coord:
                continue

            dist = self._get_sp_dist(next_coord, goal_coord)
            if dist < sp_dist:
                sp_dist = dist
                sp_action = a
        return sp_action

    def _get_action_sp_dists(self,
                             coord: Coord,
                             prev_coord: Coord,
                             goal_coord: Coord) -> List[float]:
        dists = []
        for a in self._action_space:
            new_coord = self._grid.get_next_coord(coord, a, False)
            if new_coord is None or new_coord == prev_coord:
                # action leaves position unchanged or reverses direction
                # so penalize distance by setting it to max possible dist
                d = self._get_sp_dist(coord, goal_coord)
                d += 2
            else:
                d = self._get_sp_dist(new_coord, goal_coord)
            dists.append(d)
        return dists

    def _get_sp_dist(self, coord: Coord, goal_coord: Coord) -> float:
        return self._dists[goal_coord][coord]

    def _get_coord_and_target_coord(self,
                                    history: Optional[AgentHistory]
                                    ) -> Tuple[Coord, Coord, Coord]:
        if history is None:
            return self._coord, self._prev_coord, self._target_coord
        _, obs = history.get_last_step()
        coord, prev_coord = self._get_coord_from_history(history)
        target_coord = obs[3] if self._is_evader else obs[1]
        return coord, prev_coord, target_coord

    def _get_coord_from_history(self,
                                history: AgentHistory
                                ) -> Tuple[Coord, Coord]:
        _, obs = history.get_last_step()
        coord = obs[1] if self._is_evader else obs[2]
        prev_coord = coord
        for (a, _) in history:
            if a is None:
                continue
            prev_coord = coord
            coord = self._grid.get_next_coord(coord, a, False)
        return coord, prev_coord
