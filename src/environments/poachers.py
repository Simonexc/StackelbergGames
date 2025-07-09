import uuid
from typing import TYPE_CHECKING
from functools import cached_property

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


class PoachersMap(Data):
    """
    PoachersMap: Undirected graph-based environment with node- and action-level rewards and costs,
    randomly generated using Watts-Strogatz-like topology logic.
    """
    MEAN_DEGREE = 4
    MAX_DEGREE = 4
    REWIRING_PROBABILITY = 0.1
    PERCENTAGE_ENTRY_NODES = 0.2
    PERCENTAGE_REWARD_NODES = 0.2
    MIN_ENTRY_NODES = 2
    MIN_REWARD_NODES = 2

    def __init__(self, config: "EnvConfig", device: torch.device | str | None = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config is None:
            super().__init__()
            return

        if config.seed is not None:
            generator = torch.Generator().manual_seed(config.seed)
        else:
            generator = torch.Generator()

        edge_index = self._generate_graph(config.num_nodes, generator, device)
        entry_nodes = self._create_binary_nodes(self.MIN_ENTRY_NODES, self.PERCENTAGE_ENTRY_NODES, config.num_nodes, generator, device)

        reward_nodes = self._create_binary_nodes(self.MIN_REWARD_NODES, self.PERCENTAGE_REWARD_NODES, config.num_nodes, generator, device, exclude_nodes=torch.where(entry_nodes)[0])

        node_rewards = torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device)
        node_rewards[~reward_nodes] = 0.0  # Set rewards to 0 for non-reward nodes
        upgrade_cost = -torch.rand(config.num_nodes, generator=generator, dtype=torch.float32).to(device)
        upgrade_cost[~reward_nodes] = 0.0  # Set upgrade costs to 0 for non-reward nodes
        move_cost = -torch.rand(torch.Size(()), generator=generator, dtype=torch.float32).to(device) / config.num_nodes
        preparation_reward = 1 + torch.rand(torch.Size(()), generator=generator, dtype=torch.float32).to(device)  # Fixed preparation reward

        super().__init__(
            x=torch.stack([node_rewards, upgrade_cost], dim=1),
            edge_index=edge_index,
            entry_nodes=entry_nodes,
            reward_nodes=reward_nodes,
            move_cost=move_cost,
            preparation_reward=preparation_reward,
            device=device,
        )

    @classmethod
    def load(cls, path: str, device: torch.device) -> "PoachersMap":
        obj = torch.load(path, weights_only=False).to(device)
        obj.device = device
        return obj

    @property
    def entry_nodes_list(self) -> torch.Tensor:
        """
        Returns a tensor indicating which nodes are entry nodes.
        """
        return torch.where(self.entry_nodes)[0]

    def save(self, path: str) -> None:
        torch.save(self, path)

    def get_neighbors(self, nodes: torch.Tensor) -> torch.Tensor:
        mask = self.edge_index[0].unsqueeze(0) == nodes.unsqueeze(1)
        output = torch.full((nodes.shape[0], 4), -1, dtype=torch.int64, device=self.device)
        for i, row in enumerate(mask):
            current_neighbors = self.edge_index[1][row].unique()
            assert current_neighbors.shape[0] <= 4
            output[i, :current_neighbors.shape[0]] = current_neighbors
        return output

    def _generate_graph(self, num_nodes: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
        edge_list = []
        degrees = torch.zeros(num_nodes, dtype=torch.int32)
        for i in range(num_nodes):
            for j in range(1, self.MEAN_DEGREE // 2 + 1):
                if degrees[i] >= self.MAX_DEGREE:
                    break
                neighbor = (i + j) % num_nodes
                if torch.rand(torch.Size(()), generator=generator) < self.REWIRING_PROBABILITY:
                    new_neighbor = torch.randint(
                        low=0,
                        high=num_nodes,
                        size=torch.Size(()),
                        generator=generator,
                    ).item()
                    if degrees[new_neighbor] >= self.MAX_DEGREE or new_neighbor == i:
                        continue
                    edge_list.extend([(i, new_neighbor), (new_neighbor, i)])
                    degrees[i] += 1
                    degrees[new_neighbor] += 1

                else:
                    if degrees[neighbor] >= self.MAX_DEGREE or neighbor == i:
                        continue
                    edge_list.extend([(i, neighbor), (neighbor, i)])
                    degrees[i] += 1
                    degrees[neighbor] += 1

        # Remove duplicates and self-loops
        edge_set = set((u, v) for u, v in edge_list if u != v)
        edge_index = torch.tensor(list(edge_set), dtype=torch.long, device=device).t().contiguous()  # convert to shape [2, num_edges]

        return edge_index

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

    @staticmethod
    def _create_binary_nodes(min_num: int, percentage: float, num_nodes: int, generator: torch.Generator, device: torch.device, exclude_nodes: torch.Tensor | None = None) -> torch.Tensor:
        """
        Create a binary tensor indicating which nodes are selected based on the given percentage.
        Ensures at least `min_num` nodes are selected.
        """
        if exclude_nodes is None:
            exclude_nodes = torch.tensor([], dtype=torch.int32, device=device)

        num_selected = max(min_num, int(num_nodes * percentage))
        selected_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        perm = torch.randperm(num_nodes, generator=generator).to(device)
        perm = perm[~torch.isin(perm, exclude_nodes)]  # Exclude specified nodes
        selected_nodes[perm[:num_selected]] = True

        return selected_nodes


class PoachersEnv(EnvBase):
    INVALID_MOVE_PENALTY = -500

    def __init__(self, config: "EnvConfig", poachers_map: PoachersMap, device: torch.device | str | None = None, batch_size: torch.Size | None = None) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if batch_size is None:
            batch_size = torch.Size([])

        assert batch_size == torch.Size([]), "Batch size must be a empty."

        self.map = poachers_map.to(device)
        self.num_steps = config.num_steps

        super().__init__(device=device, batch_size=batch_size)
        self._make_spec()
        assert isinstance(self.action_spec, Bounded), "Action shape should be of type Bounded."

        # Set initial positions to two random entry nodes.
        self.position = self.map.entry_nodes_list[torch.randperm(self.map.entry_nodes_list.shape[-1], device=self.device)[:2]]
        self.nodes_prepared = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)
        self.nodes_collected = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)

        self.step_count = torch.zeros((*self.batch_size, 1), dtype=torch.int32, device=self.device)
        # Initialize game_id tensor placeholder; actual UUIDs will be set in _reset
        self.game_id = torch.empty((*self.batch_size, 16), dtype=torch.uint8, device=self.device)

    def _set_seed(self, seed: int | None) -> None:
        """
        We don't need the seed here since all of the operations are deterministic.
        """

    @property
    def action_size(self) -> int:
        action_size = self.action_spec.high.unique()
        assert action_size.numel() == 1, f"Action spec high should be a single value, got {action_size}."
        return action_size.item() + 1  # +1 because high is exclusive

    def _get_action_mask(self, positions: torch.Tensor) -> torch.Tensor:
        available_moves = self.map.get_neighbors(positions)
        action_mask = torch.full((*self.batch_size, 2, 4 + 3), False, dtype=torch.bool, device=self.device)  # 4 move actions + track + prepare + collect
        action_mask[..., :4] = available_moves != -1  # Move actions
        action_mask[..., 4] = True  # Track action
        action_mask[..., 1, 5] = (
            self.map.reward_nodes[positions[1]]
            & ~self.nodes_prepared[positions[1]]
            & ~self.nodes_collected[positions[1]]
        )  # Prepare action
        action_mask[..., 1, 6] = (
            self.map.reward_nodes[positions[1]]
            & ~self.nodes_collected[positions[1]]
        )  # Collect action

        return action_mask

    def _get_graph_x(self, track_value: torch.Tensor) -> torch.Tensor:
        # node_features = self.map.x.clone()

        defender = torch.cat([
            #node_features,
            self.map.reward_nodes.float().unsqueeze(-1),
            torch.where(torch.arange(self.map.num_nodes, device=self.device) == self.position[0].item(), self.nodes_prepared.float(), -1.0).unsqueeze(-1),
            torch.where(torch.arange(self.map.num_nodes, device=self.device) == self.position[0].item(), self.nodes_collected.float(), -1.0).unsqueeze(-1),
            torch.nn.functional.one_hot(self.position[0], num_classes=self.map.num_nodes).float().unsqueeze(-1),
            track_value[0].float().unsqueeze(-1) / self.num_steps,
        ], dim=-1)

        attacker = torch.cat([
            #node_features[:, 0].unsqueeze(-1) / 10,
            #node_features[:, 1].unsqueeze(-1),
            self.map.reward_nodes.float().unsqueeze(-1),
            torch.where(torch.arange(self.map.num_nodes, device=self.device) == self.position[1].item(), self.nodes_prepared.float(), -1.0).unsqueeze(-1),
            torch.where(torch.arange(self.map.num_nodes, device=self.device) == self.position[1].item(), self.nodes_collected.float(), -1.0).unsqueeze(-1),
            torch.nn.functional.one_hot(self.position[1], num_classes=self.map.num_nodes).float().unsqueeze(-1),
            track_value[1].float().unsqueeze(-1) / self.num_steps,
        ], dim=-1)

        return torch.stack([defender, attacker], dim=-3)

    def _make_spec(self) -> None:
        self.state_spec = Composite(
            {
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
        node_features_size = 7

        self.observation_spec = Composite(
            {
                "graph_x": Unbounded(
                    shape=torch.Size((*self.batch_size, 2, self.map.num_nodes, node_features_size)),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "graph_edge_index": Bounded(
                    low=0,
                    high=self.map.num_nodes - 1,
                    shape=torch.Size((*self.batch_size, *self.map.edge_index.shape)),
                    dtype=torch.int64,
                    device=self.device,
                ),
                "track_value": Bounded(
                    low=-self.num_steps,  # -1 means unvisited
                    high=self.num_steps,
                    shape=torch.Size((*self.batch_size, 2, self.num_steps+1, self.map.num_nodes)),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "step_count_seq": Bounded(
                    low=-1,  # -1 means unobserved
                    high=self.num_steps,
                    shape=torch.Size((*self.batch_size, self.num_steps+1)),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "position_seq": Bounded(
                    low=-1,  # -1 means not valid
                    high=self.map.num_nodes - 1,
                    shape=torch.Size((*self.batch_size, 2, self.num_steps+1)),
                    dtype=torch.int64,
                    device=self.device,
                ),
                "available_moves": Bounded(
                    low=-1,  # -1 means that this move is not available
                    high=self.map.num_nodes - 1,
                    shape=torch.Size((*self.batch_size, 2, self.num_steps+1, 4)),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "node_reward_info": Bounded(
                    low=-1,  # -1 means no reward info
                    high=1,  # 1 means that this node has given information
                    shape=torch.Size((*self.batch_size, 2, self.num_steps+1, 2)),  # 2: prepared, collected
                    dtype=torch.int32,
                    device=self.device,
                ),
                "actions_mask": Bounded(
                    low=0,
                    high=1,
                    shape=torch.Size((*self.batch_size, 2, 4 + 3)),
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
                "actions_seq": Bounded(
                    low=-1,  # -1 means no action
                    high=4 + 3 - 1,  # 4 move actions + track + prepare + collect
                    shape=torch.Size((*self.batch_size, 2, self.num_steps+1)),
                    dtype=torch.int32,
                    device=self.device,
                )
            },
            shape=self.batch_size,
            device=self.device,
        )

        self.action_spec = Bounded(
            low=0,
            high=4 + 3 - 1,
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
                # "done": Bounded(
                #     low=0,
                #     high=1,
                #     shape=torch.Size((*self.batch_size, 1)),
                #     dtype=torch.bool,
                #     device=self.device,
                # ),
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

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        self.position = self.map.entry_nodes_list[torch.randperm(self.map.entry_nodes_list.shape[-1], device=self.device)[:2]]
        position_seq = torch.full((*self.batch_size, 2, self.num_steps+1), -1, dtype=torch.int64, device=self.device)
        position_seq[..., -1] = self.position.clone()

        self.step_count = torch.zeros((*self.batch_size, 1), dtype=torch.int32, device=self.device)
        step_count_seq = torch.full((*self.batch_size, self.num_steps+1), -1, dtype=torch.int32, device=self.device)
        step_count_seq[..., -1] = 0  # Set the last step count to 0

        available_moves = torch.full((*self.batch_size, 2, self.num_steps+1, 4), -1, dtype=torch.int32, device=self.device)
        available_moves[..., -1, :] = self.map.get_neighbors(self.position).int()

        # Generate game_id for each item in the batch
        if self.batch_size == torch.Size([]):  # Single environment case
            self.game_id = torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8, device=self.device)
        else:  # Batched environment case
            batch_uuids = [torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8, device=self.device) for _ in
                           range(self.batch_size[0])]
            self.game_id = torch.stack(batch_uuids, dim=0)

        track_value = torch.full((*self.batch_size, 2, self.num_steps+1, self.map.num_nodes), -self.num_steps, dtype=torch.int32, device=self.device)
        self.nodes_collected = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)
        self.nodes_prepared = torch.zeros(self.map.num_nodes, dtype=torch.bool, device=self.device)

        return TensorDict({
            "position_seq": position_seq,
            "track_value": track_value,
            "available_moves": available_moves,
            "graph_x": self._get_graph_x(track_value[..., -1, :]),
            "graph_edge_index": self.map.edge_index.clone(),
            "node_reward_info": torch.full((*self.batch_size, 2, self.num_steps+1, 2), -1, dtype=torch.int32, device=self.device),  # 2: prepared, collected
            "actions_seq": torch.full((*self.batch_size, 2, self.num_steps+1), -1, dtype=torch.int32, device=self.device),
            "actions_mask": self._get_action_mask(self.position),
            "step_count": self.step_count.clone(),
            "step_count_seq": step_count_seq,
            "game_id": self.game_id.clone(),
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict["action"]
        previous_position_seq = tensordict["position_seq"][..., 1:]
        previous_track_value = tensordict["track_value"][..., 1:, :]
        previous_available_moves = tensordict["available_moves"][..., 1:, :]
        previous_node_reward_info = tensordict["node_reward_info"][..., 1:, :]
        previous_step_count_seq = tensordict["step_count_seq"][..., 1:]
        previous_actions_seq = tensordict["actions_seq"][..., 1:]
        batch_shape = self.batch_size

        rewards = torch.zeros((*batch_shape, 2), dtype=torch.float32, device=self.device)
        is_truncated = torch.zeros((1,), dtype=torch.bool, device=self.device)
        is_terminated = torch.zeros((1,), dtype=torch.bool, device=self.device)

        # Move
        move_mask = actions < 4
        if move_mask.any():
            move_actions = actions[move_mask]
            move_positions = self.map.get_neighbors(self.position[move_mask])
            new_positions = move_positions[torch.arange(move_positions.shape[0]), move_actions]
            self.position[move_mask] = new_positions
            is_terminated[0] |= (self.position == -1).any()  # Check if any position is invalid
            rewards[move_mask] += self.map.move_cost  # * self.map.distances_to_nearest_reward[new_positions] / self.map.max_distance
            rewards[self.position == -1] += self.INVALID_MOVE_PENALTY  # Apply penalty for invalid moves

        # Check if capture
        if self.position[0].item() == self.position[1].item():
            reward_add = self.map.x[self.map.reward_nodes & ~self.nodes_collected, 0].sum()  # Defender collects all uncollected rewards
            rewards[0] += reward_add
            rewards[1] -= reward_add  # Attacker loses the not collected rewards
            is_terminated[0] = True

        # Track
        track_mask = actions == 4
        new_track = torch.full((2, 1, self.map.num_nodes), -self.num_steps, dtype=torch.int32, device=self.device)
        if track_mask.any() and not is_terminated[0].item() and not is_truncated[0].item():
            when_visited = torch.stack(
                [
                    torch.max(torch.where(previous_position_seq[..., 1, :] == self.position[0].item(), torch.arange(self.num_steps).to(self.device), -1)),
                    torch.max(torch.where(previous_position_seq[..., 0, :] == self.position[1].item(), torch.arange(self.num_steps).to(self.device), -1)),
                ],
                dim=-1,
            ).int()
            new_track[track_mask, 0, self.position[track_mask]] = when_visited[track_mask]

        attacker_node = self.position[1].item()
        is_reward_node = self.map.reward_nodes[attacker_node].item()
        # Prepare
        prepare_mask = actions == 5
        if prepare_mask.any() and not is_terminated[0].item() and not is_truncated[0].item():
            can_be_prepared = (
                prepare_mask[1].item()
                and is_reward_node
                and (not self.nodes_prepared[attacker_node].item())
                and (not self.nodes_collected[attacker_node].item())
            )
            if can_be_prepared:
                self.nodes_prepared[attacker_node] = True
                rewards[prepare_mask] += self.map.x[attacker_node, 1]  # Upgrade cost
            else:
                rewards[1] += self.INVALID_MOVE_PENALTY
                is_terminated[0] = True

        # Collect
        collect_mask = actions == 6
        if collect_mask.any() and not is_terminated[0].item() and not is_truncated[0].item():
            can_be_collected = (
                collect_mask[1].item()
                and is_reward_node
                and (not self.nodes_collected[attacker_node].item())
            )
            if can_be_collected:
                self.nodes_collected[attacker_node] = True
                if self.nodes_prepared[attacker_node].item():
                    rewards[1] += self.map.preparation_reward
                    rewards[0] -= self.map.preparation_reward  # Defender loses the preparation reward
                rewards[1] += self.map.x[attacker_node, 0]  # Node reward
                rewards[0] -= self.map.x[attacker_node, 0]
            else:
                rewards[1] += self.INVALID_MOVE_PENALTY
                is_terminated[0] = True

        # Check if all reward nodes are collected
        if self.nodes_collected[self.map.reward_nodes].all():
            is_terminated[0] = True

        reward_info = torch.full((2, 2), -1, dtype=torch.int32, device=self.device)  # 3: reward, prepared, collected
        reward_info_mask = self.map.reward_nodes[self.position]
        reward_info[reward_info_mask, 0] = self.nodes_prepared[self.position[reward_info_mask]].int()  # Prepared
        reward_info[reward_info_mask, 1] = self.nodes_collected[self.position[reward_info_mask]].int()  # Collected

        self.step_count += 1
        if self.step_count >= self.num_steps:
            is_truncated[0] = True

        return TensorDict({
            "track_value": torch.cat([
                previous_track_value,
                new_track,
            ], dim=-2),
            "step_count_seq": torch.cat([previous_step_count_seq, self.step_count.clone()], dim=-1),
            "position_seq": torch.cat([
                previous_position_seq,
                self.position.unsqueeze(-1),
            ], dim=-1),
            "available_moves": torch.cat([
                previous_available_moves,
                self.map.get_neighbors(self.position).unsqueeze(-2).int(),
            ], dim=-2),
            "graph_x": self._get_graph_x(new_track.squeeze(-2)),
            "graph_edge_index": self.map.edge_index.clone(),
            "node_reward_info": torch.cat([
                previous_node_reward_info,
                reward_info.unsqueeze(-2),
            ], dim=-2),
            "actions_seq": torch.cat([previous_actions_seq, actions.unsqueeze(-1)], dim=-1),
            "actions_mask": self._get_action_mask(self.position),
            "step_count": self.step_count.clone(),
            "reward": rewards,
            # "done": is_terminated | is_truncated,
            "truncated": is_truncated,
            "terminated": is_terminated,
        }, batch_size=self.batch_size, device=self.device)
