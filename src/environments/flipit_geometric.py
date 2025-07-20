import uuid
from typing import TYPE_CHECKING

from torch_geometric.data import Data
import torch
from torchrl.data import (
    Bounded,
    Unbounded,
    Composite,
)
from tensordict.base import TensorDictBase
from tensordict import TensorDict
from torchrl.envs import EnvBase

if TYPE_CHECKING:
    from config import EnvConfig


class FlipItMap(Data):
    """
    FlipItMap: Directed graph-based environment with node-level rewards and costs,
    randomly generated using Watts-Strogatz-like topology logic.
    """
    MEAN_DEGREE = 4
    REWIRING_PROBABILITY = 0.1
    PERCENTAGE_ENTRY_NODES = 0.2
    MIN_ENTRY_NODES = 2

    def __init__(self, config: "EnvConfig", device: torch.device | str | None = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.seed is not None:
            generator = torch.Generator().manual_seed(config.seed)
        else:
            generator = torch.Generator()

        edge_index, entry_nodes = self._generate_graph(config.num_nodes, generator, device)

        node_rewards = torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device)
        node_costs = -torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device)

        super().__init__(
            x=torch.stack([node_rewards, node_costs], dim=1),
            edge_index=edge_index,
            entry_nodes=entry_nodes,
            device=device,
        )

    @classmethod
    def load(cls, path: str, device: torch.device) -> "FlipItMap":
        obj = torch.load(path, weights_only=False).to(device)
        obj.device = device
        return obj

    def save(self, path: str) -> None:
        torch.save(self, path)

    def get_successors(self, node_idx: int) -> torch.Tensor:
        """
        Return a boolean mask indicating which nodes are successors of the given node.
        """
        mask = self.edge_index[0] == node_idx
        is_successor = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        is_successor[self.edge_index[1][mask]] = True
        return is_successor

    def get_successors_batched(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean (*node_indices_shape, num_nodes) mask indicating which nodes are successors of the given nodes.
        """
        mask = self.edge_index[0] == node_indices.unsqueeze(-1)
        non_zero = mask.nonzero(as_tuple=True)
        batch_node_indices = non_zero[:-1]
        edge_indices = non_zero[-1]
        successor_nodes = self.edge_index[1][edge_indices]

        successors_matrix = torch.zeros((*node_indices.shape, self.num_nodes), dtype=torch.bool, device=self.device)
        successors_matrix[*batch_node_indices, successor_nodes] = True
        return successors_matrix

    def get_predecessors(self, node_idx: int) -> torch.Tensor:
        """
        Return a boolean mask indicating which nodes are predecessors of the given node.
        """
        mask = self.edge_index[1] == node_idx
        is_predecessor = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
        is_predecessor[self.edge_index[0][mask]] = True
        return is_predecessor

    def get_predecessors_batched(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Return a boolean (*node_indices_shape, num_nodes) mask indicating which nodes are predecessors of the given nodes.
        """
        mask = self.edge_index[1] == node_indices.unsqueeze(-1)
        non_zero = mask.nonzero(as_tuple=True)
        batch_node_indices = non_zero[:-1]
        edge_indices = non_zero[-1]
        predecessor_nodes = self.edge_index[0][edge_indices]

        predecessor_matrix = torch.zeros((*node_indices.shape, self.num_nodes), dtype=torch.bool, device=self.device)
        predecessor_matrix[*batch_node_indices, predecessor_nodes] = True
        return predecessor_matrix

    def _generate_graph(self, num_nodes: int, generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        edge_list = []
        for i in range(num_nodes):
            for j in range(1, self.MEAN_DEGREE // 2 + 1):
                neighbor = (i + j) % num_nodes
                if torch.rand(torch.Size(()), generator=generator) < self.REWIRING_PROBABILITY:
                    new_neighbor = torch.randint(
                        low=0,
                        high=num_nodes,
                        size=torch.Size(()),
                        generator=generator,
                    ).item()
                    if new_neighbor != i:
                        direction = [(i, new_neighbor), (new_neighbor, i)][torch.randint(
                            low=0,
                            high=2,
                            size=torch.Size(()),
                            generator=generator,
                        ).item()]
                        edge_list.append(direction)

                else:
                    direction = [(i, neighbor), (neighbor, i)][
                        torch.randint(
                            low=0,
                            high=2,
                            size=torch.Size(()),
                            generator=generator,
                        ).item()
                    ]
                    edge_list.append(direction)

        # Remove duplicates and self-loops
        edge_set = set((u, v) for u, v in edge_list if u != v)
        edge_index = torch.tensor(list(edge_set), dtype=torch.long, device=device).t().contiguous()  # convert to shape [2, num_edges]

        # Mark certain percentage of nodes as entry points
        num_entry = max(self.MIN_ENTRY_NODES, int(num_nodes * self.PERCENTAGE_ENTRY_NODES))
        entry_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        entry_nodes[torch.randperm(num_nodes, generator=generator).to(device)[:num_entry]] = True

        return edge_index, entry_nodes


class FlipItEnv(EnvBase):
    def __init__(self, config: "EnvConfig", flip_it_map: FlipItMap, device: torch.device | str | None = None, batch_size: torch.Size | None = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if batch_size is None:
            batch_size = torch.Size([])

        self.map = flip_it_map.to(device)
        self.num_steps = config.num_steps

        super().__init__(device=device, batch_size=batch_size)
        self._make_spec()
        assert isinstance(self.action_spec, Bounded), "Action shape should be of type Bounded."

        self.node_owners = torch.zeros((*self.batch_size, self.map.num_nodes), dtype=torch.bool, device=self.device)
        self.step_count = torch.zeros((*self.batch_size, 1), dtype=torch.int32, device=self.device)
        # Initialize game_id tensor placeholder; actual UUIDs will be set in _reset
        self.game_id = torch.empty((*self.batch_size, 16), dtype=torch.uint8, device=self.device)

    def _make_spec(self) -> None:
        self.state_spec = Composite(
            {
                "node_owners": Bounded(
                    low=0,
                    high=1,
                    shape=torch.Size((*self.batch_size, self.map.num_nodes,)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "step_count": Bounded(
                    low=0,
                    high=self.num_steps,
                    shape=torch.Size((*self.batch_size, 1)),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "game_id": Unbounded(
                    shape=torch.Size((*self.batch_size, 16)), # UUID as 16 bytes
                    dtype=torch.uint8,
                    device=self.device,
                ),
            },
            shape=self.batch_size,
            device=self.device,
        )

        self.observation_spec = Composite(
            {
                "observed_node_owners": Bounded(
                    low=-1,  # -1 means unobserved
                    high=1,
                    shape=torch.Size((*self.batch_size, 2, self.num_steps+1, self.map.num_nodes)),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "node_owners": Bounded(
                    low=0,
                    high=1,
                    shape=torch.Size((*self.batch_size, self.map.num_nodes)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "step_count_seq": Bounded(
                    low=-1,  # -1 means unobserved
                    high=self.num_steps,
                    shape=torch.Size((*self.batch_size, self.num_steps+1)),
                    dtype=torch.int32,
                    device=self.device,
                ),

                "step_count": Bounded(
                    low=0,
                    high=self.num_steps,
                    shape=torch.Size((*self.batch_size, 1)),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "actions_seq": Bounded(
                    low=-1,
                    high=2*self.map.num_nodes - 1,  # -1 means unobserved
                    shape=torch.Size((*self.batch_size, 2, self.num_steps+1)),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "actions_mask": Bounded(
                    low=0,
                    high=1,
                    shape=torch.Size((*self.batch_size, 2, self.map.num_nodes * 2)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            shape=self.batch_size,
            device=self.device,
        )

        self.action_spec = Bounded(
            low=0,
            high=(2*self.map.num_nodes) - 1,
            shape=torch.Size((*self.batch_size, 2)),
            dtype=torch.int32,
            device=self.device,
        )

        self.reward_spec = Unbounded(
            shape=torch.Size((*self.batch_size, 2)),
            dtype=torch.float32,
            device=self.device,
        )

        self.done_spec = Composite(
            {
                "done": Bounded(
                    low=0,
                    high=1,
                    shape=torch.Size((*self.batch_size, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "truncated": Bounded(
                    low=0,
                    high=1,
                    shape=torch.Size((*self.batch_size, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                "terminated": Bounded(
                    low=0,
                    high=1,
                    shape=torch.Size((*self.batch_size, 1)),
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            shape=self.batch_size,
            device=self.device,
        )

    @property
    def action_size(self) -> int:
        action_size = self.action_spec.high.unique()
        assert action_size.numel() == 1, f"Action spec high should be a single value, got {action_size}."
        return action_size.item() + 1  # +1 because high is exclusive

    def _set_seed(self, seed: int | None) -> None:
        """
        We don't need the seed here since all of the operations are deterministic.
        """

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.node_owners = torch.zeros((*self.batch_size, self.map.num_nodes), dtype=torch.bool, device=self.device)
        self.step_count = torch.zeros((*self.batch_size, 1), dtype=torch.int32, device=self.device)
        step_count_seq = torch.full((*self.batch_size, self.num_steps+1), -1, dtype=torch.int32, device=self.device)
        step_count_seq[..., -1] = 0  # Set the last step count to 0

        # Generate game_id for each item in the batch
        if self.batch_size == torch.Size([]):  # Single environment case
            self.game_id = torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8, device=self.device)
        else:  # Batched environment case
            batch_uuids = [torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8, device=self.device) for _ in
                           range(self.batch_size[0])]
            self.game_id = torch.stack(batch_uuids, dim=0)

        return TensorDict({
            "node_owners": self.node_owners.clone(),
            "observed_node_owners": torch.full((*self.batch_size, 2, self.num_steps+1, self.map.num_nodes), -1, dtype=torch.int32, device=self.device),
            "actions_seq": torch.full((*self.batch_size, 2, self.num_steps+1), -1, dtype=torch.int32, device=self.device),
            "step_count": self.step_count.clone(),
            "step_count_seq": step_count_seq,
            "game_id": self.game_id.clone(),
            "actions_mask": torch.ones((*self.batch_size, 2, self.map.num_nodes * 2), dtype=torch.bool, device=self.device),
        }, batch_size=self.batch_size, device=self.device)

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

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict["action"]
        previous_observed_node_owners = tensordict["observed_node_owners"][..., 1:, :]
        previous_step_count_seq = tensordict["step_count_seq"][..., 1:]
        previous_actions_seq = tensordict["actions_seq"][..., 1:]
        batch_shape = self.batch_size

        defender_action_types = actions[..., 0] // self.map.num_nodes
        defender_target_nodes = actions[..., 0] % self.map.num_nodes
        attacker_action_types = actions[..., 1] // self.map.num_nodes
        attacker_target_nodes = actions[..., 1] % self.map.num_nodes

        # --- Identify Flip Attempts and Calculate Costs ---
        is_defender_flip = (defender_action_types == 0)
        is_attacker_flip = (attacker_action_types == 0)
        defender_flip_costs = torch.gather(self.map.x[..., 1].expand(*batch_shape, self.map.num_nodes), -1,
                                           defender_target_nodes.to(torch.int64)).squeeze(-1)
        attacker_flip_costs = torch.gather(self.map.x[..., 1].expand(*batch_shape, self.map.num_nodes), -1,
                                           attacker_target_nodes.to(torch.int64)).squeeze(-1)

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
                valid_defender_flip &
                valid_attacker_flip &
                (defender_target_nodes == attacker_target_nodes)
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

        step_rewards = torch.stack([defender_reward, attacker_reward], dim=-1)  # (*batch_size, 2)
        # --- Calculate Total Reward (Rewards + Costs) ---
        total_step_rewards = step_rewards + step_costs  # (*batch_size, 2)

        # --- Update Step Count and Check Termination ---
        self.step_count += 1
        is_truncated = torch.zeros_like(self.step_count, dtype=torch.bool)  # No truncation in this environment
        is_terminated = self.step_count >= self.num_steps  # (*batch_size,)
        is_done = is_terminated  # Game only ends via timeout (but it is not truncation)

        # --- Prepare Output TensorDict ---
        return TensorDict({
            "observed_node_owners": torch.cat([previous_observed_node_owners, observation_values.unsqueeze(-2)], dim=-2),
            "node_owners": self.node_owners.clone(),
            "step_count_seq": torch.cat([previous_step_count_seq, self.step_count.clone()], dim=-1),
            "actions_seq": torch.cat([previous_actions_seq, actions.unsqueeze(-1)], dim=-1),
            "step_count": self.step_count.clone(),
            "reward": total_step_rewards,
            "done": is_done,
            "truncated": is_truncated,
            "terminated": is_terminated,
            "actions_mask": torch.ones((*self.batch_size, 2, self.map.num_nodes * 2), dtype=torch.bool, device=self.device),
        }, batch_size=self.batch_size, device=self.device)
