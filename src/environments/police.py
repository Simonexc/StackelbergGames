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


class PoliceMap(EnvMapUndirected):
    """
    PoliceMap: Undirected graph-based environment representing a city grid
    where police officers (defenders) try to catch a robber (attacker).
    """
    MAX_DEGREE = 4
    PERCENTAGE_TARGET_NODES = 0.3  # Percentage of nodes that are targets
    MIN_TARGET_NODES = 3  # Minimum number of target nodes

    def _prepare_x_and_kwargs(self, config: "EnvConfig", generator: torch.Generator, device: torch.device | str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_nodes = self._create_binary_nodes(
            self.MIN_TARGET_NODES,
            self.PERCENTAGE_TARGET_NODES,
            config.num_nodes,
            generator,
            device,
        )

        # Target rewards - random rewards for each target
        target_rewards = torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device) * 10.0
        target_rewards[~target_nodes] = 0.0  # Zero reward for non-target nodes

        return (
            target_rewards.unsqueeze(-1),  # Node features: just rewards
            {
                "target_nodes": target_nodes,
            }
        )

    @property
    def target_nodes_list(self) -> torch.Tensor:
        """Returns indices of target nodes."""
        return torch.where(self.target_nodes)[0]

    def get_neighbors(self, nodes: torch.Tensor) -> torch.Tensor:
        """Get neighbors for given nodes, padded to max 4 neighbors."""
        mask = self.edge_index[0].unsqueeze(0) == nodes.unsqueeze(1)
        output = torch.full((nodes.shape[0], 4), -1, dtype=torch.int64, device=self.device)
        for i, row in enumerate(mask):
            current_neighbors = self.edge_index[1][row].unique()
            assert current_neighbors.shape[0] <= 4
            output[i, :current_neighbors.shape[0]] = current_neighbors
        return output


class PoliceEnv(EnvironmentBase):
    def __init__(self, config: "EnvConfig", env_map: PoliceMap, device: torch.device | str | None = None, batch_size: torch.Size | None = None, freeze_start_point: bool = False) -> None:
        super().__init__(config, env_map, device, batch_size, num_defenders=2)#3)

        self.see_past_points = 5  # Defenders can see if attacker visited in last k turns

        self.freeze_start_point = freeze_start_point

        self.position = torch.full((self.num_defenders + 1,), -1, dtype=torch.long, device=self.device)  # Last position of each player
        if self.freeze_start_point:
            self._generator.manual_seed(123)

        # Get hideout index
        perm = torch.randperm(self.map.num_nodes, generator=self._generator, device=self.device)
        perm = perm[~torch.isin(perm, self.map.target_nodes_list)]
        self.hideout_idx = perm[0].item()
        self.position[-1] = self.hideout_idx

        # Get police starting positions
        perm = torch.randperm(self.map.num_nodes, generator=self._generator, device=self.device)
        perm = perm[~torch.isin(perm, self.map.target_nodes_list)]
        perm = perm[perm != self.hideout_idx]
        self.position[:-1] = perm[:self.num_defenders]

        # Track game state
        self.targets_attacked = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)
        self.can_attack = True
        self.attacker_history = torch.full((self.num_steps + 1, ), -1, dtype=torch.int32, device=self.device)
        self.attacker_history[0] = self.position[-1].item()  # Attacker's initial position
        self.currently_holding_reward = 0.0

    @property
    def action_size(self) -> int:
        # Attacker: 4 move + 1 attack
        # Defender: 4 move + 5 arrest (current + 4 neighbors) + 5 check (current + 4 neighbors)
        return 14

    @property
    def graph_x_size(self) -> int:
        return 6+self.num_defenders  # Features per node

    @property
    def actions_mask(self) -> torch.Tensor:
        action_mask = torch.full((*self.batch_size, self.num_defenders + 1, self.action_size), False, dtype=torch.bool, device=self.device)

        available_moves = self.map.get_neighbors(self.position)
        action_mask[..., :4] = available_moves != -1

        # Can only attack if: on a target node, target not attacked yet, and can_attack is True
        if self.map.target_nodes[self.position[-1]].item() and not self.targets_attacked[self.position[-1]].item() and self.can_attack:
            action_mask[..., -1, 4] = True  # Attack action

        action_mask[..., :-1, [4, 9]] = True  # Arrest current position and Check current position always possible
        action_mask[..., :-1, 5:9] = available_moves[:-1] != -1  # Arrest neighbors
        action_mask[..., :-1, 10:] = available_moves[:-1] != -1  # Check neighbors

        return action_mask

    def _get_observation_spec(self) -> dict[str, TensorSpec]:
        return {
            "position_seq": Bounded(
                low=-1,
                high=self.map.num_nodes - 1,
                shape=torch.Size((*self.batch_size, self.num_defenders + 1, self.num_steps + 1)),
                dtype=torch.int64,
                device=self.device,
            ),
            "available_moves": Bounded(
                low=-1,
                high=self.map.num_nodes - 1,
                shape=torch.Size((*self.batch_size, self.num_defenders + 1, self.num_steps + 1, 4)),
                dtype=torch.int32,
                device=self.device,
            ),
            "targets_attacked_obs": Bounded(
                low=-1,  # -1 means unknown
                high=1,   # 1 means attacked
                shape=torch.Size((*self.batch_size, 2, self.num_steps + 1, self.map.num_nodes)),  # All defenders have same view
                dtype=torch.int32,
                device=self.device,
            ),
            "check_results": Bounded(
                low=-1,  # -1 means no check performed
                high=self.num_steps,  # Value indicates when attacker visited last (step number)
                shape=torch.Size((*self.batch_size, self.num_steps + 1, self.map.num_nodes)),
                dtype=torch.int32,
                device=self.device,
            ),
            "can_attack_obs": Bounded(
                low=0,
                high=1,
                shape=torch.Size((*self.batch_size, self.num_steps + 1)),
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def _get_graph_x(self, **kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate graph features for GNN input."""
        check_results = kwargs["check_results"][..., -1, :]

        # All defenders
        defender_features = torch.stack([
            self.map.x.squeeze(-1) / 10,  # Target rewards
            self.map.target_nodes.float(),  # Is target node
            torch.zeros(self.map.num_nodes, dtype=torch.float),  # Is hideout - defender doesn't know
            # Targets already attacked - only visible if any defender is on it
            torch.where(torch.isin(torch.arange(self.map.num_nodes, device=self.device), self.position[:-1]), self.targets_attacked.float(), -1.0),
            check_results.float(),
        ] + [
            (torch.arange(self.map.num_nodes, device=self.device) == self.position[i]).float() for i in range(self.num_defenders)
        ] + [
            torch.zeros(self.map.num_nodes, device=self.device, dtype=torch.float)  # Attacker position - defender doesn't know
        ], dim=-1)

        # For attacker (player -1)
        attacker_features = torch.stack([
            self.map.x.squeeze(-1),  # Target rewards
            self.map.target_nodes.float(),  # Is target node
            (torch.arange(self.map.num_nodes, device=self.device) == self.hideout_idx).float(),  # Is hideout
            self.targets_attacked.float(),  # Targets already attacked
            torch.zeros_like(check_results).float(),
        ] + [
            (torch.arange(self.map.num_nodes, device=self.device) == self.position[i]).float() for i in range(self.num_defenders + 1)
        ], dim=-1)

        return torch.stack([defender_features, attacker_features], dim=-3)

    def _impl_reset(self) -> dict[str, torch.Tensor]:
        if self.freeze_start_point:
            self._generator.manual_seed(123)

        # Get hideout index
        perm = torch.randperm(self.map.num_nodes, generator=self._generator, device=self.device)
        perm = perm[~torch.isin(perm, self.map.target_nodes_list)]
        self.hideout_idx = perm[0].item()
        self.position[-1] = self.hideout_idx

        # Get police starting positions
        perm = torch.randperm(self.map.num_nodes, generator=self._generator, device=self.device)
        perm = perm[~torch.isin(perm, self.map.target_nodes_list)]
        perm = perm[perm != self.hideout_idx]
        self.position[:-1] = perm[:self.num_defenders]

        self.targets_attacked = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)
        self.can_attack = True
        self.attacker_history = torch.full((self.num_steps + 1, ), -1, dtype=torch.int32, device=self.device)
        self.attacker_history[0] = self.position[-1].item()  # Attacker's initial position
        self.currently_holding_reward = 0.0

        # Initialize observation tensors
        position_seq = torch.full((*self.batch_size, self.num_defenders + 1, self.num_steps + 1), -1, dtype=torch.int64, device=self.device)
        position_seq[..., -1] = self.position.clone()

        available_moves = torch.full((*self.batch_size, self.num_defenders + 1, self.num_steps + 1, 4), -1, dtype=torch.int32, device=self.device)
        available_moves[..., -1, :] = self.map.get_neighbors(self.position).int()

        targets_attacked_obs = torch.full((*self.batch_size, 2, self.num_steps + 1, self.map.num_nodes), -1, dtype=torch.int32, device=self.device)
        # Attacker knows all attacked targets
        targets_attacked_obs[..., 1, -1, :] = self.targets_attacked.int()
        targets_attacked_obs[..., 0, -1, :] = torch.where(torch.isin(torch.arange(self.map.num_nodes, device=self.device), self.position[:-1]), self.targets_attacked.int(), -1)

        check_results = torch.full((*self.batch_size, self.num_steps + 1, self.map.num_nodes), -1, dtype=torch.int32, device=self.device)

        can_attack_obs = torch.full((*self.batch_size, self.num_steps + 1), False, dtype=torch.bool, device=self.device)
        can_attack_obs[..., -1] = self.can_attack

        return {
            "position_seq": position_seq,
            "available_moves": available_moves,
            "targets_attacked_obs": targets_attacked_obs,
            "check_results": check_results,
            "can_attack_obs": can_attack_obs,
        }

    def _impl_step(
        self, tensordict: TensorDictBase, rewards: torch.Tensor, is_truncated: torch.Tensor, is_terminated: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        actions = tensordict["action"]
        # Previous observations
        prev_position_seq = tensordict["position_seq"][..., 1:]
        prev_available_moves = tensordict["available_moves"][..., 1:, :]
        prev_targets_attacked_obs = tensordict["targets_attacked_obs"][..., 1:, :]
        prev_check_results = tensordict["check_results"][..., 1:, :]
        prev_can_attack_obs = tensordict["can_attack_obs"][..., 1:]

        # Move
        move_mask = actions < 4
        if move_mask.any():
            move_actions = actions[move_mask]
            move_positions = self.map.get_neighbors(self.position[move_mask])
            new_positions = move_positions[torch.arange(move_positions.shape[0]), move_actions]
            self.position[move_mask] = new_positions
            if (self.position == -1).any():
                raise ValueError(f"Invalid action detected: {actions}")

        if self.position[-1] == self.hideout_idx and not self.can_attack:
            self.can_attack = True
            rewards[1] += self.currently_holding_reward * 0.8  # Attacker gets 80% of held reward on returning to hideout
            rewards[0] -= self.currently_holding_reward  # Defenders lose that reward
            self.currently_holding_reward = 0.0

        # Update attacker history
        self.attacker_history[self.step_count+1] = self.position[-1].item()

        # Attack
        attacker_action = actions[-1]
        if attacker_action == 4:
            if self.can_attack and self.map.target_nodes[self.position[-1]] and not self.targets_attacked[self.position[-1]]:
                self.targets_attacked[self.position[-1]] = True
                rewards[1] += self.map.x[self.position[-1], 0].item() * 0.2  # Attacker gets 20% of target reward on collection
                self.can_attack = False
                self.currently_holding_reward = self.map.x[self.position[-1], 0].item()
            else:
                raise ValueError(f"Invalid attack action detected: {actions}")
                # rewards[..., -1] -= 100.0

        # Arrest
        arrest_mask = (4 <= actions) & (actions < 9)
        arrest_mask[-1] = False  # Last action is attacker
        if arrest_mask.any():
            arrest_actions = actions[arrest_mask]
            possible_arrest_positions = torch.cat([self.position[arrest_mask].unsqueeze(-1), self.map.get_neighbors(self.position[arrest_mask])], dim=-1)
            arrest_positions = possible_arrest_positions[torch.arange(possible_arrest_positions.shape[0]), arrest_actions - 4]
            if (arrest_positions == -1).any():
                raise ValueError(f"Invalid arrest action detected: {actions}")
            else:
                if (self.position[-1] == arrest_positions).any():
                    # Caught the attacker!
                    uncollected_reward = self.map.x[self.map.target_nodes & ~self.targets_attacked, 0].sum().item()
                    rewards[0] += uncollected_reward + self.currently_holding_reward  # Defenders win
                    rewards[1] -= uncollected_reward + self.currently_holding_reward * 0.2  # Attacker loses
                    is_terminated[0] = True

        # Check node
        check_mask = 9 <= actions
        check_mask[-1] = False  # Last action is attacker
        check_results = torch.full((self.map.num_nodes,), -1, dtype=torch.int32, device=self.device)
        if check_mask.any():
            check_actions = actions[check_mask]
            possible_check_positions = torch.cat([self.position[check_mask].unsqueeze(-1), self.map.get_neighbors(self.position[check_mask])], dim=-1)
            check_positions = possible_check_positions[torch.arange(possible_check_positions.shape[0]), check_actions - 9]
            if (check_positions == -1).any():
                raise ValueError(f"Invalid check action detected: {actions}")
            else:
                check_results[check_positions] = torch.isin(check_positions, self.attacker_history[max(0, self.step_count - self.see_past_points+1):self.step_count+1]).int()

        # Check if all targets collected and attacker returned to hideout
        if self.targets_attacked[self.map.target_nodes].all() and self.can_attack:
            is_terminated[0] = True

        # Targets attacked observation
        new_targets_attacked_obs = torch.stack([
            torch.where(torch.isin(torch.arange(self.map.num_nodes, device=self.device), self.position[:-1]), self.targets_attacked.int(), -1),
            self.targets_attacked.int(),
        ], dim=-2)

        return {
            "position_seq": torch.cat([
                prev_position_seq,
                self.position.unsqueeze(-1),
            ], dim=-1),
            "available_moves": torch.cat([
                prev_available_moves,
                self.map.get_neighbors(self.position).unsqueeze(-2).int(),
            ], dim=-2),
            "targets_attacked_obs": torch.cat([
                prev_targets_attacked_obs, new_targets_attacked_obs.unsqueeze(-2)
            ], dim=-2),
            "check_results": torch.cat([prev_check_results, check_results.unsqueeze(-2)], dim=-2),
            "can_attack_obs": torch.cat([prev_can_attack_obs, torch.tensor([self.can_attack])], dim=-1),
        }
