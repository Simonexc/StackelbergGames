from enum import Enum
from typing import NamedTuple

import torch
import random

from .flipit_geometric import FlipItEnv
from config import Player


class BeliefState:
    def __init__(self, player: Player, env: FlipItEnv, believed_node_owners: torch.Tensor | None = None) -> None:
        self.believed_node_owners = torch.full((env.map.num_nodes,), Player.defender.value, dtype=torch.bool) if believed_node_owners is None else believed_node_owners
        self.beliefs_history: list[tuple[Player, PlayerTargetPair] | None] = []
        self.player = player
        self.env = env

    @classmethod
    def from_observation_history(cls, player: Player, env: FlipItEnv, observation_history: list[PlayerTargetPair | None], believed_node_owners: torch.Tensor | None = None) -> "BeliefState":
        instance = cls(player, env, believed_node_owners)
        for observation in observation_history:
            if observation is not None:
                instance.update_belief(observation)
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

    def nodes_reachable(self) -> list[int]:
        """
        Get a list of nodes that are reachable by the player.

        Returns:
            list[int]: A list of reachable nodes.
        """
        return [node for node in range(self.env.map.num_nodes) if self.is_believed_reachable(node)]

    def update_belief(self, target: PlayerTargetPair | None) -> None:
        if target is None:
            self.beliefs_history.append(None)
            return
        self.beliefs_history.append((Player(self.believed_node_owners[target.target_node]), target))
        self.believed_node_owners[target.target_node] = target.player.value

    def undo_belief(self) -> None:
        if len(self.beliefs_history) <= 0:
            raise RuntimeError("No belief to undo.")

        last_belief = self.beliefs_history.pop()
        if last_belief is None:
            return
        self.believed_node_owners[last_belief[1].target_node] = last_belief[0].value


def generate_random_pure_strategy(player: Player, env: FlipItEnv) -> torch.Tensor:
    """
    Generate a random pure strategy for the FlipIt game.

    Args:
        num_steps (int): Number of steps in the game.
        num_nodes (int): Number of nodes in the game graph.

    Returns:
        list[ActionTargetPair]: A list of action-target pairs representing the pure strategy.
    """

    pure_strategy: list[int] = []
    belief_state = BeliefState(player, env)
    for step in range(env.num_steps):
        target_node = random.choice(belief_state.nodes_reachable())
        pure_strategy.append(target_node)
        belief_state.update_belief(PlayerTargetPair(player=player, target_node=target_node))

    return torch.tensor(pure_strategy, dtype=torch.int32)


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
