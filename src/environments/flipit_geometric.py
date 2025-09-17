from typing import TYPE_CHECKING

import torch
from torchrl.data import (
    Bounded,
    TensorSpec,
)
from tensordict.base import TensorDictBase

from .base_map import EnvMapDirected
from .base_env import EnvironmentBase

if TYPE_CHECKING:
    from config import EnvConfig


class FlipItMap(EnvMapDirected):
    """
    FlipItMap: Directed graph-based environment with node-level rewards and costs.
    """
    PERCENTAGE_ENTRY_NODES = 0.2
    MIN_ENTRY_NODES = 2

    def _prepare_x_and_kwargs(self, config: "EnvConfig", generator: torch.Generator, device: torch.device | str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        node_rewards = torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device)
        node_costs = -torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device)

        return (
            torch.stack([node_rewards, node_costs], dim=1),
            {
                "entry_nodes": self._create_binary_nodes(
                    self.MIN_ENTRY_NODES, self.PERCENTAGE_ENTRY_NODES, config.num_nodes, generator, device
                ),
            },
        )


class FlipItEnv(EnvironmentBase):
    def __init__(self, config: "EnvConfig", env_map: FlipItMap, device: torch.device | str | None = None, batch_size: torch.Size | None = None) -> None:
        super().__init__(config, env_map, device, batch_size)

        self.node_owners = torch.zeros((*self.batch_size, self.map.num_nodes), dtype=torch.bool, device=self.device)

    @property
    def action_size(self) -> int:
        return self.map.num_nodes * 2  # Defender and attacker can flip or observe each node

    @property
    def graph_x_size(self) -> int:
        return 3

    @property
    def actions_mask(self) -> torch.Tensor:
        return torch.ones((*self.batch_size, 2, self.map.num_nodes * 2), dtype=torch.bool, device=self.device)

    def _get_observation_spec(self) -> dict[str, TensorSpec]:
        return {
            "observed_node_owners": Bounded(
                low=-1,  # -1 means unobserved
                high=1,
                shape=torch.Size((*self.batch_size, 2, self.num_steps + 1, self.map.num_nodes)),
                dtype=torch.int32,
                device=self.device,
            ),
            "node_owners_fi": Bounded(  # fi - full information; used in oracle agents
                low=0,
                high=1,
                shape=torch.Size((*self.batch_size, self.map.num_nodes)),
                dtype=torch.bool,
                device=self.device,
            ),

        }

    def _impl_reset(self) -> dict[str, torch.Tensor]:
        self.node_owners = torch.zeros((*self.batch_size, self.map.num_nodes), dtype=torch.bool, device=self.device)

        return {
            "node_owners_fi": self.node_owners.clone(),
            "observed_node_owners": torch.full(
                (*self.batch_size, 2, self.num_steps + 1, self.map.num_nodes),
                -1,
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def is_reachable_for_defender_batched(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Check if the defender can reach the target nodes.
        """
        return (
            ~self.node_owners.expand((*node_indices.shape, self.map.num_nodes))  # defender is the owner
            & self.map.get_successors_batched(node_indices).to(self.device)  # the node's successors
        ).any(dim=1)

    def is_reachable_for_attacker_batched(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Check if the attacker can reach the target nodes.
        """
        return (
            self.node_owners.expand((*node_indices.shape, self.map.num_nodes))  # attacker is the owner
            & self.map.get_predecessors_batched(node_indices).to(self.device)  # the node's predecessors
        ).any(dim=1) | self.map.entry_nodes.to(self.device)[node_indices]  # is entry node

    def _get_graph_x(self, **kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        observed_node_owners = kwargs["observed_node_owners"][..., -1, :]
        node_features = self.map.x.clone()
        defender = torch.cat([
            node_features,
            observed_node_owners[..., 0, :].unsqueeze(-1).to(torch.float32),
        ], dim=-1)

        attacker = torch.cat([
            node_features,
            observed_node_owners[..., 1, :].unsqueeze(-1).to(torch.float32),
        ], dim=-1)

        return torch.stack([defender, attacker], dim=-3)


    def _impl_step(
        self, tensordict: TensorDictBase, rewards: torch.Tensor, is_truncated: torch.Tensor, is_terminated: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        actions = tensordict["action"]
        previous_observed_node_owners = tensordict["observed_node_owners"][..., 1:, :]

        defender_action_types = actions[..., 0] // self.map.num_nodes
        defender_target_nodes = actions[..., 0] % self.map.num_nodes
        attacker_action_types = actions[..., 1] // self.map.num_nodes
        attacker_target_nodes = actions[..., 1] % self.map.num_nodes

        # --- Identify Flip Attempts and Calculate Costs ---
        is_defender_flip = (defender_action_types == 0)
        is_attacker_flip = (attacker_action_types == 0)
        defender_flip_costs = torch.gather(
            self.map.x[..., 1].expand(*self.batch_size, self.map.num_nodes),
            -1,
            defender_target_nodes.to(torch.int64),
        ).squeeze(-1)
        attacker_flip_costs = torch.gather(
            self.map.x[..., 1].expand(*self.batch_size, self.map.num_nodes),
            -1,
            attacker_target_nodes.to(torch.int64),
        ).squeeze(-1)

        step_costs = torch.stack(
            [
                torch.where(is_defender_flip, defender_flip_costs, 0.0),
                torch.where(is_attacker_flip, attacker_flip_costs, 0.0),
            ],
            dim=-1,
        )

        # --- Determine Valid Flips (Reachability) ---
        can_defender_flip = torch.zeros_like(is_defender_flip)
        can_defender_flip[is_defender_flip] = self.is_reachable_for_defender_batched(
            defender_target_nodes[is_defender_flip]
        )

        can_attacker_flip = torch.zeros_like(is_attacker_flip)
        can_attacker_flip[is_attacker_flip] = self.is_reachable_for_attacker_batched(
            attacker_target_nodes[is_attacker_flip]
        )

        valid_defender_flip = is_defender_flip & can_defender_flip
        valid_attacker_flip = is_attacker_flip & can_attacker_flip

        # --- Handle Flip Conflicts (Rule 1.c) ---
        # Conflict occurs if both attempt a valid flip on the *same* node
        conflict = (
                valid_defender_flip
                & valid_attacker_flip
                & (defender_target_nodes == attacker_target_nodes)
        )

        # Ignore flips if there is a conflict
        execute_defender_flip = valid_defender_flip & ~conflict
        execute_attacker_flip = valid_attacker_flip & ~conflict

        # --- Update Node Ownership ---
        # Create tensors for updates: value to write (0 or 1) and mask for which nodes to update
        flip_values = torch.zeros_like(self.node_owners)  # Value to scatter
        flip_mask = torch.zeros_like(self.node_owners)  # Mask indicating where to scatter

        # Prepare defender updates
        defender_targets = defender_target_nodes[execute_defender_flip]
        # Need to index flip_mask/values using batch indices where execute_defender_flip is true
        #batch_idx_def = execute_defender_flip.nonzero(as_tuple=True)
        #flip_mask[*batch_idx_def, defender_targets] = True
        flip_mask[defender_targets] = True
        # flip_values already 0 where needed

        # Prepare attacker updates
        attacker_targets = attacker_target_nodes[execute_attacker_flip]
        #batch_idx_att = execute_attacker_flip.nonzero(as_tuple=True)
        #flip_mask[*batch_idx_att, attacker_targets] = True
        #flip_values[*batch_idx_att, attacker_targets] = True  # Set to attacker owner
        flip_mask[attacker_targets] = True
        flip_values[attacker_targets] = True  # Set to attacker owner

        # Apply updates using the mask
        self.node_owners = torch.where(flip_mask, flip_values, self.node_owners)

        # --- Perform Observations ---
        defender_observation_indexes = (defender_action_types == 1).nonzero(as_tuple=True)[0]
        attacker_observation_indexes = (attacker_action_types == 1).nonzero(as_tuple=True)[0]

        observation_values = torch.full((*self.batch_size, 2, self.map.num_nodes), -1, dtype=torch.int32, device=self.device)

        defender_target_nodes_observed = defender_target_nodes.unsqueeze(-1)[defender_observation_indexes]
        #observation_values[*defender_observation_indexes, 0, defender_target_nodes_observed] = self.node_owners[*defender_observation_indexes, defender_target_nodes_observed].to(torch.int32)
        observation_values[0, defender_target_nodes_observed] = self.node_owners[defender_target_nodes_observed].to(torch.int32)

        attacker_target_nodes_observed = attacker_target_nodes.unsqueeze(-1)[attacker_observation_indexes]
        #observation_values[*attacker_observation_indexes, 1, attacker_target_nodes_observed] = self.node_owners[*attacker_observation_indexes, attacker_target_nodes_observed].to(torch.int32)
        observation_values[1, attacker_target_nodes_observed] = self.node_owners[attacker_target_nodes_observed].to(torch.int32)

        # --- Calculate Step Rewards (Based on new ownership) ---
        # node_rewards shape (n,) -> expand to (*batch_size, n)
        #batch_node_rewards = self.map.x[..., 0].expand(*batch_shape, self.map.num_nodes)
        batch_node_rewards = self.map.x[..., 0].expand(self.map.num_nodes)

        # Defender reward: sum of rewards for nodes they own (~self._node_owners)
        defender_reward = torch.sum(batch_node_rewards * (~self.node_owners), dim=-1)  # (*batch_size,)

        # Attacker reward: sum of rewards for nodes they own (self._node_owners)
        attacker_reward = torch.sum(batch_node_rewards * self.node_owners, dim=-1)  # (*batch_size,)

        rewards[..., 0] = defender_reward
        rewards[..., 1] = attacker_reward
        rewards += step_costs

        return {
            "observed_node_owners": torch.cat([previous_observed_node_owners, observation_values.unsqueeze(-2)], dim=-2),
            "node_owners_fi": self.node_owners.clone(),
        }
