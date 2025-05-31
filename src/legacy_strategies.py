from abc import ABC, abstractmethod
from typing import Callable, Deque
from copy import deepcopy
from itertools import product
from collections import deque

import numpy as np
import torch

from environments.flipit import FlipItEnv
from environments.flipit_utils import BeliefState, Action, ActionTargetPair, PlayerTargetPair, PlayerActionTargetPair
from config import Player


class Policy(ABC):
    """
    Abstract base class for policies.
    """
    def __init__(self, action_space: list[ActionTargetPair]) -> None:
        self.action_space = action_space

        self.observation_history: list[PlayerTargetPair | None] = []

    def _extend_observations(self, extended_observations: list[PlayerTargetPair | None] | None) -> list[PlayerTargetPair | None]:
        """
        Extend the observation history with the given observations.

        Args:
            extended_observations (list[PlayerTargetPair | None] | None): The observations to extend.

        Returns:
            list[PlayerTargetPair | None]: The extended observations.
        """
        if extended_observations is None:
            return self.observation_history

        return self.observation_history + extended_observations

    @abstractmethod
    def get_probs(self, extended_observations: list[PlayerTargetPair | None] | None = None) -> list[float]:
        """
        Get the probabilities of actions given the current history of observations with optional extensions.

        Returns a list of probabilities for each action in the action space.
        """

    def sample_action(self, extended_observations: list[PlayerTargetPair | None] | None = None) -> ActionTargetPair:
        """
        Sample an action based on the probabilities.

        Returns a sampled action from the action space.
        """
        probs = self.get_probs(extended_observations)
        assert np.isclose(sum(probs), 1.0), f"Probabilities must sum to 1.0, but got {sum(probs)}"
        action_index = np.random.choice(len(self.action_space), p=probs)
        return self.action_space[action_index]

    def add_observation(self, observation: PlayerTargetPair | None) -> None:
        """
        Add an observation to the history.

        Args:
            observation (PlayerTargetPair | None): The observation to add.
        """
        self.observation_history.append(observation)


class AttackerPolicy(Policy, ABC):
    def __init__(self, action_space: list[ActionTargetPair], defender_policy: Policy) -> None:
        super().__init__(action_space)
        self.defender_policy = defender_policy


class OracleAttackerPolicy(AttackerPolicy):
    """
    Oracle policy for the attacker.
    """

    def __init__(self, action_space: list[ActionTargetPair], defender_policy: Policy, env: FlipItEnv) -> None:
        super().__init__(action_space, defender_policy)
        self.env = env

    def get_probs(self, extended_observations: list[PlayerTargetPair | None] | None = None) -> list[float]:
        base_full_observations = self._extend_observations(extended_observations)
        m = len(base_full_observations)

        defender_extended_observations: list[PlayerTargetPair | None] = []
        defender_probs = self.defender_policy.get_probs()
        belief_state = BeliefState(Player.attacker, self.env, self.env.node_owners.clone())

        reachable = belief_state.reachable_attacker_fast()
        og_reasonable_action_space = [action_target for action_target in self.action_space if (reachable[action_target.target_node] and action_target.action == Action.flip) or (action_target.action == Action.observe and action_target.target_node == 0)]
        # (defender, attacker), (cumulative_reward_defender, cumulative_reward_attacker), org_attacker_index, relative_step, prob
        queue: Deque[tuple[tuple[ActionTargetPair, ActionTargetPair], tuple[float, float], int, int, float]] = deque(
            ((defender_action_target, attacker_action_target), (0, 0), i, 0, prob)
            for (defender_action_target, prob), (i, attacker_action_target) in product(zip(self.action_space, defender_probs), enumerate(og_reasonable_action_space))
            if prob > 1e-9
        )
        weighted_score = (
            torch.tensor([action_target.action.value for action_target in og_reasonable_action_space], dtype=torch.int32),
            torch.tensor([action_target.target_node for action_target in og_reasonable_action_space], dtype=torch.int32),
            torch.full((len(og_reasonable_action_space), 2), -float("inf"), dtype=torch.float32),
        )

        operations_count = 0
        while len(queue) > 0:
            operations_count += 1
            (defender_action_target, attacker_action_target), (cumulative_reward_defender, cumulative_reward_attacker), original_action_target, relative_step, prob = queue.pop()
            defender_extended_observations = defender_extended_observations[:relative_step]

            while len(belief_state.beliefs_history) > 2*relative_step:
                belief_state.undo_belief()
                belief_state.undo_belief()
                self.env.undo_step()

            _, rewards, _, _ = self.env.step(defender_action_target, attacker_action_target)
            #print(cumulative_reward_defender, cumulative_reward_attacker, rewards)
            cumulative_reward_defender += rewards[Player.defender.value].detach().item()
            cumulative_reward_attacker += rewards[Player.attacker.value].detach().item()

            # Logic to update the belief state
            if defender_action_target.target_node == attacker_action_target.target_node:
                if defender_action_target.action == Action.flip:
                    if attacker_action_target.action == Action.flip:
                        belief_state.update_belief(None)
                        belief_state.update_belief(None)
                    else:
                        belief_state.update_belief(PlayerTargetPair(Player.defender, defender_action_target.target_node))
                        belief_state.update_belief(None)
                else:
                    if attacker_action_target.action == Action.flip:
                        belief_state.update_belief(None)
                        belief_state.update_belief(PlayerTargetPair(Player.attacker, attacker_action_target.target_node))
                    else:
                        belief_state.update_belief(None)
                        belief_state.update_belief(None)
            else:
                if defender_action_target.action == Action.flip:
                    belief_state.update_belief(PlayerTargetPair(Player.defender, defender_action_target.target_node))
                else:
                    belief_state.update_belief(None)
                if attacker_action_target.action == Action.flip:
                    belief_state.update_belief(PlayerTargetPair(Player.attacker, attacker_action_target.target_node))
                else:
                    belief_state.update_belief(None)

            if defender_action_target.action == Action.observe:
                defender_extended_observations.append(PlayerTargetPair(Player(belief_state.believed_node_owners[defender_action_target.target_node]), defender_action_target.target_node))

            if relative_step + m + 1 >= self.env.m:
                reward = torch.tensor([cumulative_reward_defender, cumulative_reward_attacker], dtype=torch.float32) * prob
                #print(reward)
                if torch.isinf(weighted_score[2][original_action_target]).any():
                    weighted_score[2][original_action_target] = reward
                else:
                    weighted_score[2][original_action_target] += reward
                continue

            defender_probs = self.defender_policy.get_probs(defender_extended_observations)
            reachable = belief_state.reachable_attacker_fast()
            reasonable_action_space = [action_target for action_target in self.action_space if (reachable[action_target.target_node] and action_target.action == Action.flip) or (action_target.action == Action.observe and action_target.target_node == 0)]
            for (defender_action_target, prob_act), attacker_action_target in product(zip(self.action_space, defender_probs), reasonable_action_space):
                new_prob = prob * prob_act * 1/len(og_reasonable_action_space)
                if new_prob < 1e-9:
                    continue
                #print(cumulative_reward_defender, cumulative_reward_attacker, new_prob)
                queue.append((
                    (defender_action_target, attacker_action_target),
                    (cumulative_reward_defender, cumulative_reward_attacker),
                    original_action_target,
                    relative_step + 1,
                    new_prob,
                ))
        print(operations_count)

        while len(belief_state.beliefs_history) > 0:
            belief_state.undo_belief()
            belief_state.undo_belief()
            self.env.undo_step()
        #print(weighted_score)
        max_indexes = torch.where(weighted_score[2][:, 1] == torch.max(weighted_score[2][:, 1]))[0]
        max_defender = torch.argmax(weighted_score[2][max_indexes, 0])
        action = weighted_score[0][max_indexes[max_defender]].item()
        target = weighted_score[1][max_indexes[max_defender]].item()
        #print(action, target, weighted_score[2][max_indexes[max_defender]])
        # max_indexes = torch.where(weighted_score[2] == torch.max(weighted_score[2][:, 1], dim=0).values)[0]
        # maximal_weighted_score = max(weighted_score, key=lambda x: x[2])
        # equal_attacker_scores = [
        #     (action_target, defender_score)
        #     for action_target, defender_score, attacker_score in weighted_score
        #     if np.isclose(attacker_score, maximal_weighted_score[2], rtol=1e-9, atol=1e-9)
        # ]
        #
        # best = max(equal_attacker_scores, key=lambda x: x[1])
        #
        return [1.0 if bool(at.action.value) == action and at.target_node == target else 0.0 for at in self.action_space]
