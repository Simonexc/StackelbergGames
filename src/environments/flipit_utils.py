from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, Type

import torch
import random

from .base_env import EnvironmentBase
from .base_map import EnvMapBase
from .flipit_geometric import FlipItEnv, FlipItMap
from .poachers import PoachersEnv, PoachersMap
from config import Player


class Action(Enum):
    flip = 0
    observe = 1

    @classmethod
    def random(cls) -> "Action":
        return cls(random.choice(list(cls.__members__.values())))


class ActionTargetPair(NamedTuple):
    action: Action
    target_node: int


class PlayerTargetPair(NamedTuple):
    player: Player
    target_node: int


class PlayerActionTargetPair(NamedTuple):
    player: Player
    action: Action
    target_node: int


class BeliefStateBase(ABC):
    def __init__(self, player: Player, env: EnvironmentBase, device: torch.device) -> None:
        self.player = player
        self.env = env
        self.device = device

    @abstractmethod
    def update_belief(self, action: int) -> None:
        """
        Update the belief state based on the action taken.
        """

    @abstractmethod
    def available_actions(self) -> list[int]:
        """
        Get a list of available actions for the player.
        """

    @classmethod
    @abstractmethod
    def from_actions_history(cls, player: Player, env: EnvironmentBase, actions: list[int], device: str | torch.device) -> "BeliefStateBase":
        """
        Create a belief state from a history of actions.
        """


class FlipItBeliefState(BeliefStateBase):
    def __init__(self, player: Player, env: FlipItEnv, device: torch.device, believed_node_owners: torch.Tensor | None = None) -> None:
        super().__init__(player, env, device)

        assert isinstance(env, FlipItEnv), "FlipitBeliefState can only be used with FlipItEnv."
        self.believed_node_owners = torch.full((env.map.num_nodes,), Player.defender.value, dtype=torch.bool, device=device) if believed_node_owners is None else believed_node_owners
        self.beliefs_history: list[tuple[Player, int] | None] = []

    @classmethod
    def from_actions_history(cls, player: Player, env: FlipItEnv, actions: list[int], device: str | torch.device, believed_node_owners: torch.Tensor | None = None) -> "BeliefState":
        instance = cls(player, env, device, believed_node_owners)
        for action in actions:
            instance.update_belief(action)
        return instance

    def is_believed_reachable_for_defender(self, node: int) -> bool:
        """
        Check if the believed node is reachable by the player.

        Args:
            node (int): The node to check.

        Returns:
            bool: True if the node is reachable, False otherwise.
        """
        return (self.believed_node_owners[self.env.map.get_successors(node)] == Player.defender.value).any().item()

    def is_believed_reachable_for_attacker(self, node: int) -> bool:
        """
        Check if the believed node is reachable by the player.

        Args:
            node (int): The node to check.

        Returns:
            bool: True if the node is reachable, False otherwise.
        """
        if self.env.map.entry_nodes[node]:
            return True

        return (self.env.map.get_predecessors(node) & self.believed_node_owners).any().item()

    def reachable_attacker_fast(self) -> torch.Tensor:
        return (self.env.adjacency_matrix & self.believed_node_owners).any(dim=1) | self.env.entry_nodes

    def is_believed_reachable(self, node: int) -> bool:
        """
        Check if the believed node is reachable by the player.

        Args:
            node (int): The node to check.

        Returns:
            bool: True if the node is reachable, False otherwise.
        """
        if self.player == Player.defender:
            return self.is_believed_reachable_for_defender(node)
        elif self.player == Player.attacker:
            return self.is_believed_reachable_for_attacker(node)
        else:
            raise ValueError(f"Invalid player: {self.player}.")

    def available_actions(self) -> list[int]:
        """
        Get a list of nodes that are reachable by the player.

        Returns:
            list[int]: A list of reachable nodes.
        """
        node_ids = [node for node in range(self.env.map.num_nodes) if self.is_believed_reachable(node)]
        return node_ids + [self.env.map.num_nodes + node for node in range(self.env.map.num_nodes)]  # flip and observe actions

    def update_belief(self, action: int) -> None:
        self.beliefs_history.append((Player(self.believed_node_owners[action % self.env.map.num_nodes]), action))
        self.believed_node_owners[action % self.env.map.num_nodes] = action // self.env.map.num_nodes

    # def undo_belief(self) -> None:
    #     if len(self.beliefs_history) <= 0:
    #         raise RuntimeError("No belief to undo.")
    #
    #     last_belief = self.beliefs_history.pop()
    #     if last_belief is None:
    #         return
    #     self.believed_node_owners[last_belief[1].target_node] = last_belief[0].value


class PoachersBeliefState(BeliefStateBase):
    def __init__(self, player: Player, env: PoachersEnv, device: torch.device) -> None:
        super().__init__(player, env, device)

        assert isinstance(env, PoachersEnv), "PoachersBeliefState can only be used with PoachersEnv."
        self.pos = env.position[player.value]
        self.prepared = env.nodes_prepared.clone()
        self.collected = env.nodes_collected.clone()

    def update_belief(self, action: int) -> None:
        if action < 4:
            self.pos = self.env.map.get_neighbors(torch.tensor([self.pos], device=self.env.device))[0, action]
        elif action == 5 and not self.prepared[self.pos] and not self.collected[self.pos] and self.env.map.reward_nodes[self.pos]:
            self.prepared[self.pos] = True
        elif action == 6 and not self.collected[self.pos] and self.prepared[self.pos]:
            self.collected[self.pos] = True

    def available_actions(self) -> list[int]:
        """
        Get a list of available actions for the player.

        Returns:
            list[int]: A list of available actions.
        """
        actions = []
        neighbors = self.env.map.get_neighbors(torch.tensor([self.pos], device=self.env.device))[0]
        for i, neighbor in enumerate(neighbors):
            if neighbor != -1:
                actions.append(i)
        actions.append(4)
        if self.player == Player.attacker and self.env.map.reward_nodes[self.pos]:
            if not self.prepared[self.pos] and not self.collected[self.pos]:
                actions.append(5)
            if not self.collected[self.pos]:
                actions.append(6)

        return actions

    @classmethod
    def from_actions_history(cls, player: Player, env: PoachersEnv, actions: list[int], device: str | torch.device) -> "PoachersBeliefState":
        instance = cls(player, env, device)
        for action in actions:
            instance.update_belief(action)
        return instance


def belief_state_class(env: EnvironmentBase | EnvMapBase) -> Type[BeliefStateBase]:
    if isinstance(env, FlipItEnv) or isinstance(env, FlipItMap):
        return FlipItBeliefState
    elif isinstance(env, PoachersEnv) or isinstance(env, PoachersMap):
        return PoachersBeliefState
    else:
        raise ValueError(f"Unsupported environment type: {type(env)}")


# def generate_random_pure_strategy(player: Player, env: FlipItEnv) -> torch.Tensor:
#     """
#     Generate a random pure strategy for the FlipIt game.
#
#     Args:
#         num_steps (int): Number of steps in the game.
#         num_nodes (int): Number of nodes in the game graph.
#
#     Returns:
#         list[ActionTargetPair]: A list of action-target pairs representing the pure strategy.
#     """
#
#     pure_strategy: list[int] = []
#     belief_state = BeliefState2(player, env, env.device)
#     for step in range(env.num_steps):
#         target_action = random.choice(belief_state.available_actions())
#         pure_strategy.append(target_action)
#         #belief_state.update_belief(PlayerTargetPair(player=player, target_node=target_node))
#         belief_state.update_belief(target_action)
#
#     return torch.tensor(pure_strategy, dtype=torch.int32)


def generate_random_pure_strategy(player: Player, env: EnvironmentBase) -> torch.Tensor:
    """
    Generate a random pure strategy for the FlipIt game.

    Args:
        num_steps (int): Number of steps in the game.
        num_nodes (int): Number of nodes in the game graph.

    Returns:
        list[ActionTargetPair]: A list of action-target pairs representing the pure strategy.
    """

    pure_strategy: list[int] = []
    belief_state = belief_state_class(env)(player, env, env.device)
    for step in range(env.num_steps):
        action = random.choice(belief_state.available_actions())
        pure_strategy.append(action)
        belief_state.update_belief(action)

    return torch.tensor(pure_strategy, dtype=torch.int32)
