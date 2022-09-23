"""Heuristic agents for Level-Based Foraging env.

Reference:
https://github.com/uoe-agents/lb-foraging/blob/master/lbforaging/agents/heuristic_agent.py

"""
import random
from typing import Optional, Tuple, List

import posggym.model as M
from posggym.envs.lbf.model import LBFModel
from posggym.envs.lbf.core import LBFAction
from posggym.utils.history import AgentHistory

import posggym_agents.policy as Pi


class LBFHeuristicPolicy(Pi.BaseHiddenStatePolicy):
    """Heuristic agent for the Level-Based Foraging env.

    This is the abstract Level-Based Foraging env heuristic policy class.
    Concrete implementations must implement the get_action_from_obs method.
    """

    def __init__(self,
                 model: LBFModel,
                 agent_id: M.AgentID,
                 policy_id: str):
        super().__init__(model, agent_id, policy_id)
        assert model.observation_mode in ('vector', 'tuple')

    def _get_action_from_obs(self, obs: M.Observation) -> M.Action:
        """Get action from observation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _get_pi_from_obs(self, obs: M.Observation) -> Pi.ActionDist:
        """Get action distribution from observation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_action(self) -> M.Action:
        last_obs = self.history.get_last_step()[1]
        return self._get_action_from_obs(last_obs)

    def get_action_by_hidden_state(self,
                                   hidden_state: Pi.PolicyHiddenState
                                   ) -> M.Action:
        last_obs = hidden_state["history"].get_last_step()[1]
        return self._get_action_from_obs(last_obs)

    def get_pi(self,
               history: Optional[AgentHistory] = None
               ) -> Pi.ActionDist:
        if history is None:
            history = self.history
        last_obs = history.get_last_step()[1]
        return self._get_pi_from_obs(last_obs)

    def get_pi_from_hidden_state(self,
                                 hidden_state: Pi.PolicyHiddenState
                                 ) -> Pi.ActionDist:
        last_obs = hidden_state["history"].get_last_step()[1]
        return self._get_pi_from_obs(last_obs)

    def _closest_food(self,
                      agent_pos: Tuple[int, int],
                      food_obs: List[Tuple[int, int, int]],
                      max_food_level: Optional[int] = None
                      ) -> Optional[Tuple[int, int]]:
        food_dists = {}
        for (y, x, level) in food_obs:
            if (
                x == -1
                or (max_food_level is not None and level > max_food_level)
            ):
                continue
            dist = (agent_pos[0] - y)**2 + (agent_pos[1] - x)**2
            if dist not in food_dists:
                food_dists[dist] = []
            food_dists[dist].append((y, x))

        if len(food_dists) == 0:
            # No food in sight
            return None

        return random.choice(food_dists[min(food_dists)])

    def _center_of_agents(self,
                          agent_pos: List[Tuple[int, int]]) -> Tuple[int, int]:
        y_mean = sum(coord[0] for coord in agent_pos) / len(agent_pos)
        x_mean = sum(coord[1] for coord in agent_pos) / len(agent_pos)
        return round(x_mean), round(y_mean)

    def _move_towards(self,
                      agent_pos: Tuple[int, int],
                      target: Tuple[int, int],
                      allowed_actions: List[int]) -> List[LBFAction]:
        y, x = agent_pos
        r, c = target

        valid_actions = []
        if r < y and LBFAction.NORTH in allowed_actions:
            valid_actions.append(LBFAction.NORTH)
        if r > y and LBFAction.SOUTH in allowed_actions:
            valid_actions.append(LBFAction.SOUTH)
        if c > x and LBFAction.EAST in allowed_actions:
            valid_actions.append(LBFAction.EAST)
        if c < x and LBFAction.WEST in allowed_actions:
            valid_actions.append(LBFAction.WEST)

        if valid_actions:
            return valid_actions
        else:
            raise ValueError("No simple path found")

    def _get_valid_move_actions(self,
                                agent_obs: List[Tuple[int, int]]
                                ) -> List[LBFAction]:
        y, x = agent_obs[0][:2]
        other_agent_pos = set([o[:2] for o in agent_obs[1:] if o[0] > -1])
        width, height = self.model.field_size

        valid_actions = []
        if y > 0 and (y-1, x) not in other_agent_pos:
            valid_actions.append(LBFAction.NORTH)
        if y < height-1 and (y+1, x) not in other_agent_pos:
            valid_actions.append(LBFAction.SOUTH)
        if x < width-1 and (y, x+1) not in other_agent_pos:
            valid_actions.append(LBFAction.EAST)
        if x > 0 and (y, x-1) not in other_agent_pos:
            valid_actions.append(LBFAction.WEST)

        return valid_actions

    def _get_actions_towards_food(self,
                                  agent_obs: List[Tuple[int, int, int]],
                                  center_pos: Tuple[int, int],
                                  food_obs: List[Tuple[int, int, int]],
                                  max_food_level: Optional[int] = None
                                  ) -> List[LBFAction]:
        try:
            r, c = self._closest_food(center_pos, food_obs, max_food_level)
        except TypeError:
            # from trying to unpack None
            actions = self._get_valid_move_actions(agent_obs)
            if actions:
                return actions
            return [LBFAction.NONE]

        y, x = agent_obs[0][:2]
        if (abs(r - y) + abs(c - x)) == 1:
            return [LBFAction.LOAD]

        valid_move_actions = self._get_valid_move_actions(agent_obs)
        try:
            return self._move_towards((y, x), (r, c), valid_move_actions)
        except ValueError:
            if valid_move_actions:
                return valid_move_actions
            return [LBFAction.NONE]


class LBFHeuristicPolicy1(LBFHeuristicPolicy):
    """Level-Based Foraging Heuristic Policy 1.

    This policy always goes to the closest observed food, irrespective of
    the foods level.
    """

    def _get_action_from_obs(self, obs: M.Observation) -> M.Action:
        agent_obs, food_obs = self.model.parse_obs(obs)
        agent_pos = agent_obs[0][:2]
        actions = self._get_actions_towards_food(
            agent_obs, agent_pos, food_obs
        )
        return random.choice(actions)

    def _get_pi_from_obs(self, obs: M.Observation) -> Pi.ActionDist:
        agent_obs, food_obs = self.model.parse_obs(obs)
        agent_pos = agent_obs[0][:2]
        actions = self._get_actions_towards_food(
            agent_obs, agent_pos, food_obs
        )
        action_dist = {a: 0.0 for a in LBFAction}
        for a in actions:
            action_dist[a] = 1.0 / len(actions)
        return action_dist


class LBFHeuristicPolicy2(LBFHeuristicPolicy):
    """Level-Based Foraging Heuristic Policy 2.

    This policy goes towards the visible food that is closest to the centre of
    visible players, irrespective of food level.
    """

    def _get_action_from_obs(self, obs: M.Observation) -> M.Action:
        agent_obs, food_obs = self.model.parse_obs(obs)
        other_agent_pos = [o[:2] for o in agent_obs[1:] if o[0] > -1]

        if not other_agent_pos:
            actions = self._get_valid_move_actions(agent_obs)
            if actions:
                return random.choice(actions)
            return LBFAction.NONE

        center_pos = self._center_of_agents(other_agent_pos)
        actions = self._get_actions_towards_food(
            agent_obs, center_pos, food_obs
        )
        return random.choice(actions)

    def _get_pi_from_obs(self, obs: M.Observation) -> Pi.ActionDist:
        agent_obs, food_obs = self.model.parse_obs(obs)
        other_agent_pos = [o[:2] for o in agent_obs[1:] if o[0] > -1]

        if not other_agent_pos:
            # no visible agents
            actions = self._get_valid_move_actions(agent_obs)
            if not actions:
                actions = [LBFAction.NONE]
        else:
            center_pos = self._center_of_agents(other_agent_pos)
            actions = self._get_actions_towards_food(
                agent_obs, center_pos, food_obs
            )

        action_dist = {a: 0.0 for a in LBFAction}
        for a in actions:
            action_dist[a] = 1.0 / len(actions)
        return action_dist


class LBFHeuristicPolicy3(LBFHeuristicPolicy):
    """Level-Based Foraging Heuristic Policy 3.

    This policy goes towards the closest visible food with a compatible level.
    """

    def _get_action_from_obs(self, obs: M.Observation) -> M.Action:
        agent_obs, food_obs = self.model.parse_obs(obs)
        agent_pos = agent_obs[0][:2]
        agent_level = agent_obs[0][2]
        actions = self._get_actions_towards_food(
            agent_obs, agent_pos, food_obs, agent_level
        )
        return random.choice(actions)

    def _get_pi_from_obs(self, obs: M.Observation) -> Pi.ActionDist:
        agent_obs, food_obs = self.model.parse_obs(obs)
        agent_pos = agent_obs[0][:2]
        agent_level = agent_obs[0][2]
        actions = self._get_actions_towards_food(
            agent_obs, agent_pos, food_obs, agent_level
        )
        action_dist = {a: 0.0 for a in LBFAction}
        for a in actions:
            action_dist[a] = 1.0 / len(actions)
        return action_dist


class LBFHeuristicPolicy4(LBFHeuristicPolicy):
    """Level-Based Foraging Heuristic Policy 4.

    This policy goes towards the visible food which is closest to all visible
    agents such that the sum of their and the policy agent's level is
    sufficient to load the food.
    """

    def _get_action_from_obs(self, obs: M.Observation) -> M.Action:
        agent_obs, food_obs = self.model.parse_obs(obs)
        other_agent_pos = [o[:2] for o in agent_obs[1:] if o[0] > -1]

        if not other_agent_pos:
            actions = self._get_valid_move_actions(agent_obs)
            if actions:
                return random.choice(actions)
            return LBFAction.NONE

        agent_level_sum = sum([o[2] for o in agent_obs if o[0] > -1])
        center_pos = self._center_of_agents(other_agent_pos)
        actions = self._get_actions_towards_food(
            agent_obs, center_pos, food_obs, agent_level_sum
        )
        return random.choice(actions)

    def _get_pi_from_obs(self, obs: M.Observation) -> Pi.ActionDist:
        agent_obs, food_obs = self.model.parse_obs(obs)
        other_agent_pos = [o[:2] for o in agent_obs[1:] if o[0] > -1]

        if not other_agent_pos:
            # no visible agents
            actions = self._get_valid_move_actions(agent_obs)
            if not actions:
                actions = [LBFAction.NONE]
        else:
            agent_level_sum = sum([o[2] for o in agent_obs if o[0] > -1])
            center_pos = self._center_of_agents(other_agent_pos)
            actions = self._get_actions_towards_food(
                agent_obs, center_pos, food_obs, agent_level_sum
            )

        action_dist = {a: 0.0 for a in LBFAction}
        for a in actions:
            action_dist[a] = 1.0 / len(actions)
        return action_dist
