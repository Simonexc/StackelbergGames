from typing import TYPE_CHECKING
from functools import cached_property

import torch
from torchrl.data import (
    Bounded,
    TensorSpec,
)
from tensordict.base import TensorDictBase

from .base_map import EnvMapUndirected
from .base_env import EnvironmentBase

if TYPE_CHECKING:
    from config import EnvConfig


class PoachersMap(EnvMapUndirected):
    """
    PoachersMap: Undirected graph-based environment with node- and action-level rewards and costs,
    randomly generated using Watts-Strogatz-like topology logic.
    """
    MAX_DEGREE = 4
    PERCENTAGE_ENTRY_NODES = 0.2
    PERCENTAGE_REWARD_NODES = 0.2
    MIN_ENTRY_NODES = 2
    MIN_REWARD_NODES = 2

    def _prepare_x_and_kwargs(self, config: "EnvConfig", generator: torch.Generator, device: torch.device | str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        entry_nodes = self._create_binary_nodes(
            self.MIN_ENTRY_NODES, self.PERCENTAGE_ENTRY_NODES, config.num_nodes, generator, device
        )

        reward_nodes = self._create_binary_nodes(
            self.MIN_REWARD_NODES,
            self.PERCENTAGE_REWARD_NODES,
            config.num_nodes,
            generator,
            device,
            exclude_nodes=torch.where(entry_nodes)[0]
        )

        node_rewards = 1 + torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device) * 5
        node_rewards[~reward_nodes] = 0.0  # Set rewards to 0 for non-reward nodes
        upgrade_cost = -torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device)
        upgrade_cost[~reward_nodes] = 0.0  # Set upgrade costs to 0 for non-reward nodes

        move_cost = -torch.rand(torch.Size(()), generator=generator, dtype=torch.float32).to(device) / (4 * config.num_nodes)
        preparation_reward = 1 + torch.rand(
            torch.Size(()), generator=generator, dtype=torch.float32
        ).to(device) * 2  # Fixed preparation reward

        return (
            torch.stack([node_rewards, upgrade_cost], dim=1),
            {
                "entry_nodes": entry_nodes,
                "reward_nodes": reward_nodes,
                "move_cost": move_cost,
                "preparation_reward": preparation_reward,
            }
        )

    @property
    def entry_nodes_list(self) -> torch.Tensor:
        """
        Returns a tensor indicating which nodes are entry nodes.
        """
        return torch.where(self.entry_nodes)[0]

    def get_neighbors(self, nodes: torch.Tensor) -> torch.Tensor:
        mask = self.edge_index[0].unsqueeze(0) == nodes.unsqueeze(1)
        output = torch.full((nodes.shape[0], 4), -1, dtype=torch.int64, device=self.device)
        for i, row in enumerate(mask):
            current_neighbors = self.edge_index[1][row].unique()
            assert current_neighbors.shape[0] <= 4
            output[i, :current_neighbors.shape[0]] = current_neighbors
        return output

    @cached_property
    def distances_to_nearest_reward(self) -> torch.Tensor:
        final_distances = torch.full((self.num_nodes,), float('inf'), dtype=torch.float32, device=self.device)

        nodes = torch.where(self.reward_nodes)[0].tolist()
        distances = [0] * len(nodes)
        visited: set[int] = set()

        while nodes:
            current_node = nodes.pop(0)
            distance = distances.pop(0)
            if current_node in visited or distance >= final_distances[current_node].item():
                continue
            visited.add(current_node)
            final_distances[current_node] = distance

            neighbors = self.get_neighbors(
                torch.tensor([current_node], dtype=torch.int32, device=self.device)).squeeze(0).cpu().tolist()
            for neighbor in neighbors:
                if neighbor != -1 and neighbor not in visited:
                    distances.append(distance + 1)
                    nodes.append(neighbor)

        return final_distances

    @cached_property
    def max_distance(self) -> float:
        return self.distances_to_nearest_reward.max().item()


class PoachersEnv(EnvironmentBase):
    def __init__(self, config: "EnvConfig", env_map: PoachersMap, device: torch.device | str | None = None, batch_size: torch.Size | None = None, freeze_start_point: bool = False) -> None:
        super().__init__(config, env_map, device, batch_size)

        # Set initial positions to two random entry nodes.
        if freeze_start_point:
            self.position = self.map.entry_nodes_list[:2]
        else:
            self.position = self.map.entry_nodes_list[torch.randperm(self.map.entry_nodes_list.shape[-1], generator=self._generator, device=self.device)[:2]]
        self.freeze_start_point = freeze_start_point
        self.nodes_prepared = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)
        self.nodes_collected = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)

    @property
    def action_size(self) -> int:
        return 4 + 3  # 4 move actions + track + prepare + collect

    @property
    def graph_x_size(self) -> int:
        return 7

    @property
    def actions_mask(self) -> torch.Tensor:
        available_moves = self.map.get_neighbors(self.position)
        action_mask = torch.full((*self.batch_size, 2, self.action_size), False, dtype=torch.bool, device=self.device)
        action_mask[..., :4] = available_moves != -1  # Move actions
        action_mask[..., 4] = True  # Track action
        action_mask[..., 1, 5] = (
            self.map.reward_nodes[self.position[1]]
            & ~self.nodes_prepared[self.position[1]]
            & ~self.nodes_collected[self.position[1]]
        )  # Prepare action
        action_mask[..., 1, 6] = (
            self.map.reward_nodes[self.position[1]]
            & ~self.nodes_collected[self.position[1]]
        )  # Collect action

        return action_mask

    def _get_observation_spec(self) -> dict[str, TensorSpec]:
        return {
            "track_value": Bounded(
                low=-self.num_steps,  # -1 means unvisited
                high=self.num_steps,
                shape=torch.Size((*self.batch_size, 2, self.num_steps + 1, self.map.num_nodes)),
                dtype=torch.int32,
                device=self.device,
            ),
            "position_seq": Bounded(
                low=-1,  # -1 means not valid
                high=self.map.num_nodes - 1,
                shape=torch.Size((*self.batch_size, 2, self.num_steps + 1)),
                dtype=torch.int64,
                device=self.device,
            ),
            "available_moves": Bounded(
                low=-1,  # -1 means that this move is not available
                high=self.map.num_nodes - 1,
                shape=torch.Size((*self.batch_size, 2, self.num_steps + 1, 4)),
                dtype=torch.int32,
                device=self.device,
            ),
            "node_reward_info": Bounded(
                low=-1,  # -1 means no reward info
                high=1,  # 1 means that this node has given information
                shape=torch.Size((*self.batch_size, 2, self.num_steps + 1, 2)),  # 2: prepared, collected
                dtype=torch.int32,
                device=self.device,
            ),
            "nodes_prepared_fi": Bounded(  # fi - full information; used in oracle agents
                low=0,
                high=1,
                shape=torch.Size((*self.batch_size, self.map.num_nodes)),
                dtype=torch.bool,
                device=self.device,
            ),
            "nodes_collected_fi": Bounded(  # fi - full information; used in oracle agents
                low=0,
                high=1,
                shape=torch.Size((*self.batch_size, self.map.num_nodes)),
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def _get_graph_x(self, **kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        track_value = kwargs["track_value"][..., -1, :]
        node_features = self.map.x.clone()
        defender = torch.cat([
            node_features[:, 0].unsqueeze(-1) / 5,
            node_features[:, 1].unsqueeze(-1),
            self.map.reward_nodes.float().unsqueeze(-1),
            torch.where(torch.arange(self.map.num_nodes, device=self.device) == self.position[0].item(), self.nodes_prepared.float(), -1.0).unsqueeze(-1),
            torch.where(torch.arange(self.map.num_nodes, device=self.device) == self.position[0].item(), self.nodes_collected.float(), -1.0).unsqueeze(-1),
            torch.nn.functional.one_hot(torch.max(torch.zeros_like(self.position[0]), self.position[0]), num_classes=self.map.num_nodes).float().unsqueeze(-1),
            track_value[0].float().unsqueeze(-1) / self.num_steps,
        ], dim=-1)

        attacker = torch.cat([
            node_features[:, 0].unsqueeze(-1) / 5,
            node_features[:, 1].unsqueeze(-1),
            self.map.reward_nodes.float().unsqueeze(-1),
            torch.where(torch.arange(self.map.num_nodes, device=self.device) == self.position[1].item(), self.nodes_prepared.float(), -1.0).unsqueeze(-1),
            torch.where(torch.arange(self.map.num_nodes, device=self.device) == self.position[1].item(), self.nodes_collected.float(), -1.0).unsqueeze(-1),
            torch.nn.functional.one_hot(torch.max(torch.zeros_like(self.position[1]), self.position[1]), num_classes=self.map.num_nodes).float().unsqueeze(-1),
            track_value[1].float().unsqueeze(-1) / self.num_steps,
        ], dim=-1)

        return torch.stack([defender, attacker], dim=-3)

    def _impl_reset(self) -> dict[str, torch.Tensor]:
        if self.freeze_start_point:
            self.position = self.map.entry_nodes_list[:2]
        else:
            self.position = self.map.entry_nodes_list[
                torch.randperm(self.map.entry_nodes_list.shape[-1], generator=self._generator, device=self.device)[:2]
            ]
        position_seq = torch.full((*self.batch_size, 2, self.num_steps + 1), -1, dtype=torch.int64, device=self.device)
        position_seq[..., -1] = self.position.clone()

        available_moves = torch.full((*self.batch_size, 2, self.num_steps + 1, 4), -1, dtype=torch.int32,
                                     device=self.device)
        available_moves[..., -1, :] = self.map.get_neighbors(self.position).int()

        track_value = torch.full((*self.batch_size, 2, self.num_steps + 1, self.map.num_nodes), -self.num_steps,
                                 dtype=torch.int32, device=self.device)
        self.nodes_collected = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)
        self.nodes_prepared = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)

        return {
            "position_seq": position_seq,
            "track_value": track_value,
            "available_moves": available_moves,
            "node_reward_info": torch.full(
                (*self.batch_size, 2, self.num_steps + 1, 2),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),  # 2: prepared, collected
            "nodes_collected_fi": self.nodes_collected.clone(),
            "nodes_prepared_fi": self.nodes_prepared.clone(),
        }

    def _impl_step(
        self, tensordict: TensorDictBase, rewards: torch.Tensor, is_truncated: torch.Tensor, is_terminated: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        actions = tensordict["action"]
        previous_position_seq = tensordict["position_seq"][..., 1:]
        previous_track_value = tensordict["track_value"][..., 1:, :]
        previous_available_moves = tensordict["available_moves"][..., 1:, :]
        previous_node_reward_info = tensordict["node_reward_info"][..., 1:, :]

        # Move (rule 1)
        move_mask = actions < 4
        if move_mask.any():
            move_actions = actions[move_mask]
            move_positions = self.map.get_neighbors(self.position[move_mask])
            new_positions = move_positions[torch.arange(move_positions.shape[0]), move_actions]
            self.position[move_mask] = new_positions
            if (self.position == -1).any():
                #raise ValueError(f"Invalid action detected: {actions}")
                rewards[self.position == -1] -= 100.0  # Penalty for invalid move

            rewards[move_mask] += self.map.move_cost

        # Check if capture (rule 6)
        if self.position[0].item() == self.position[1].item():
            # Defender collects all uncollected rewards
            reward_add = self.map.x[self.map.reward_nodes & ~self.nodes_collected, 0].sum()
            rewards[0] += reward_add
            rewards[1] -= reward_add  # Attacker loses the not collected rewards
            is_terminated[0] = True

        # Track (rule 2)
        track_mask = actions == 4
        new_track = torch.full((2, 1, self.map.num_nodes), -self.num_steps, dtype=torch.int32, device=self.device)
        if track_mask.any() and not is_terminated[0].item() and not is_truncated[0].item():
            when_visited = torch.stack(
                [
                    torch.max(
                        torch.where(
                            previous_position_seq[..., 1, :] == self.position[0].item(),
                            torch.arange(self.num_steps).to(self.device),
                            -1,
                        )
                    ),
                    torch.max(
                        torch.where(
                            previous_position_seq[..., 0, :] == self.position[1].item(),
                            torch.arange(self.num_steps).to(self.device),
                            -1,
                        )
                    ),
                ],
                dim=-1,
            ).int()
            new_track[track_mask, 0, self.position[track_mask]] = when_visited[track_mask]

        attackers_node = self.position[1].item()
        is_reward_node = self.map.reward_nodes[attackers_node].item()
        # Prepare
        prepare_mask = actions == 5
        if prepare_mask.any() and not is_terminated[0].item() and not is_truncated[0].item():
            can_be_prepared = (
                prepare_mask[1].item()
                and is_reward_node
                and (not self.nodes_prepared[attackers_node].item())
                and (not self.nodes_collected[attackers_node].item())
            )
            if can_be_prepared:
                self.nodes_prepared[attackers_node] = True
                rewards[prepare_mask] += self.map.x[attackers_node, 1]  # Upgrade cost
            else:
                #raise ValueError(f"Invalid action detected: {actions}. Cannot prepare node {attackers_node}.")
                rewards[prepare_mask] -= 100.0  # Penalty for invalid prepare action

        # Collect
        collect_mask = actions == 6
        if collect_mask.any() and not is_terminated[0].item() and not is_truncated[0].item():
            can_be_collected = (
                collect_mask[1].item()
                and is_reward_node
                and (not self.nodes_collected[attackers_node].item())
            )
            if can_be_collected:
                self.nodes_collected[attackers_node] = True
                if self.nodes_prepared[attackers_node].item():
                    rewards[1] += self.map.preparation_reward
                    rewards[0] -= self.map.preparation_reward  # Defender loses the preparation reward
                rewards[1] += self.map.x[attackers_node, 0]  # Node reward
                rewards[0] -= self.map.x[attackers_node, 0]
            else:
                #raise ValueError(f"Invalid action detected: {actions}. Cannot collect node {attackers_node}.")
                rewards[collect_mask] -= 100.0  # Penalty for invalid collect action

        # Check if all reward nodes are collected
        if self.nodes_collected[self.map.reward_nodes].all():
            is_terminated[0] = True

        reward_info = torch.full((2, 2), -1, dtype=torch.int32, device=self.device)  # 3: reward, prepared, collected
        reward_info_mask = self.map.reward_nodes[self.position]
        reward_info[reward_info_mask, 0] = self.nodes_prepared[self.position[reward_info_mask]].int()  # Prepared
        reward_info[reward_info_mask, 1] = self.nodes_collected[self.position[reward_info_mask]].int()  # Collected

        return {
            "track_value": torch.cat([
                previous_track_value,
                new_track,
            ], dim=-2),
            "position_seq": torch.cat([
                previous_position_seq,
                self.position.unsqueeze(-1),
            ], dim=-1),
            "available_moves": torch.cat([
                previous_available_moves,
                self.map.get_neighbors(self.position).unsqueeze(-2).int(),
            ], dim=-2),
            "node_reward_info": torch.cat([
                previous_node_reward_info,
                reward_info.unsqueeze(-2),
            ], dim=-2),
            "nodes_collected_fi": self.nodes_collected.clone(),
            "nodes_prepared_fi": self.nodes_prepared.clone(),
        }
