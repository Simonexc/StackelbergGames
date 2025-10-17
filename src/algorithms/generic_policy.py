from abc import ABC, abstractmethod
import os

import torch
import wandb
from tensordict.nn import set_interaction_type as set_exploration_type, InteractionType as ExplorationType
from tensordict.nn.probabilistic import interaction_type as get_interaction_type
from torch import nn
from tensordict import TensorDictBase
from torchrl.data import ReplayBuffer
from torchrl.envs import EnvBase

from config import Player
from .base import BaseAgent, BaseTrainableAgent
from .generator import AgentGenerator
from environments.flipit_utils import belief_state_class
from environments.base_env import EnvironmentBase


def _tensordict_update_from_action(action: torch.Tensor, embedding_size: int, action_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
    logits = torch.zeros((*action.shape, action_size), dtype=torch.float32, device=device)
    logits[torch.arange(0, action.shape[-1]), action] = 1.0
    sample_log_prob = torch.zeros_like(action, dtype=torch.float32, device=device)
    embedding = torch.zeros(embedding_size, dtype=torch.float32, device=device)
    return {
        "action": action,
        "logits": logits,
        "sample_log_prob": sample_log_prob,
        "embedding": embedding,
    }


class RandomAgent(BaseAgent):
    def __init__(
        self,
        action_size: int,
        embedding_size: int,
        player_type: int,
        device: torch.device | str,
        run_name: str,
        num_defenders: int,
        num_attackers: int,
        agent_id: int | None = None,
    ) -> None:
        super().__init__(player_type, device, run_name, num_defenders, num_attackers, agent_id)
        self.embedding_size = embedding_size
        self.action_size = action_size
        assert self.player_type == 1, "RandomAgent is only implemented for attackers."
        assert self.num_attackers == 1, "RandomAgent is only implemented for single attacker."

    def save(self) -> None:
        # RandomAgent does not save its state
        pass

    def load(self, path: str) -> None:
        # RandomAgent does not load any state
        pass

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        action_mask = tensordict["actions_mask"][..., self.num_defenders, :]
        available_actions = action_mask.nonzero(as_tuple=False).squeeze(-1)
        action = available_actions[torch.randint(0, len(available_actions), torch.Size((self.num_defenders if self.player_type == 0 else self.num_attackers,)), dtype=torch.int32).item()].unsqueeze(-1)
        tensordict.update(_tensordict_update_from_action(action, self.embedding_size, self.action_size, self._device))
        return tensordict


class CombinedPolicy(nn.Module):
    def __init__(
        self,
        defender_module: nn.Module,
        attacker_module: nn.Module,
        exploration_defender: nn.Module | None = None,
        exploration_attacker: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.defender_module = defender_module
        self.attacker_module = attacker_module
        self.exploration_defender = exploration_defender
        self.exploration_attacker = exploration_attacker

        if exploration_defender is not None or exploration_attacker is not None:
            self.exploration_coeff = 1.0
        else:
            self.exploration_coeff = 0.0

    @property
    def _defender_module(self) -> nn.Module:
        if self.exploration_defender is not None and self.exploration_coeff > 0:
            return self.exploration_defender if torch.rand(torch.Size(())) < self.exploration_coeff else self.defender_module
        return self.defender_module

    @property
    def _attacker_module(self) -> nn.Module:
        if self.exploration_attacker is not None and self.exploration_coeff > 0:
            return self.exploration_attacker if torch.rand(torch.Size(())) < self.exploration_coeff else self.attacker_module
        return self.attacker_module

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        current_device = tensordict.device
        with torch.no_grad():
            defender_output = self._defender_module(tensordict.clone()).detach().to(current_device)
            attacker_output = self._attacker_module(tensordict).detach().to(current_device)

        defender_embedding = defender_output["embedding"].clone()
        attacker_embedding = attacker_output["embedding"]
        if defender_embedding.shape[-1] != attacker_embedding.shape[-1]:
            if defender_embedding.shape[-1] < attacker_embedding.shape[-1]:
                defender_embedding = torch.cat([defender_embedding, torch.zeros_like(attacker_embedding[..., defender_embedding.shape[-1]:])], dim=-1)
            else:
                attacker_embedding = torch.cat([attacker_embedding, torch.zeros_like(defender_embedding[..., attacker_embedding.shape[-1]:])], dim=-1)

        attacker_output.update({
            "action": torch.cat([defender_output["action"].clone(), attacker_output["action"]], dim=-1),
            "logits": torch.cat([defender_output["logits"].clone(), attacker_output["logits"]], dim=-2),
            "sample_log_prob": torch.cat([defender_output["sample_log_prob"].clone(), attacker_output["sample_log_prob"]], dim=-1),
            "embedding": torch.stack([defender_embedding, attacker_embedding], dim=-1),
        })
        del defender_output
        return attacker_output

    def evaluate(self, env: EnvBase, rollout_num: int, current_player: int, add_logs: bool = True) -> torch.Tensor:
        _exploration_coeff = self.exploration_coeff
        self.exploration_coeff = 0.0  # Disable exploration during evaluation
        self.eval()
        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
            eval_rollout = env.rollout(rollout_num, self, break_when_any_done=False)

            if add_logs:
                player = self.defender_module if current_player == 0 else self.attacker_module
                wandb.log({
                    "eval/reward_defender_mean": eval_rollout["next", "reward"][..., 0].mean().item(),
                    "eval/reward_defender_std": eval_rollout["next", "reward"][..., 0].std().item(),
                    "eval/reward_attacker_mean": eval_rollout["next", "reward"][..., 1].mean().item(),
                    "eval/reward_attacker_std": eval_rollout["next", "reward"][..., 1].std().item(),
                    f"general/step_{player.player_name}": player.num_steps,
                    f"general/epoch_{player.player_name}": player.num_epochs,
                    f"general/cycle_{player.player_name}": player.num_cycle,
                })

        reward = eval_rollout["next", "reward"].clone()
        del eval_rollout
        self.exploration_coeff = _exploration_coeff

        return reward

    def single_run(self, env: EnvBase, current_player: int, add_logs: bool = True, exploration_type = ExplorationType.DETERMINISTIC) -> torch.Tensor:
        _exploration_coeff = self.exploration_coeff
        self.eval()
        with set_exploration_type(exploration_type), torch.no_grad():
            eval_rollout = env.rollout(env.num_steps, self, break_when_any_done=True)

            if add_logs:
                player = self.defender_module if current_player == 0 else self.attacker_module
                wandb.log({
                    "eval/reward_defender_single_run": eval_rollout["next", "reward"][..., 0].sum().item(),
                    "eval/reward_attacker_single_run": eval_rollout["next", "reward"][..., 1].sum().item(),
                    f"general/step_{player.player_name}": player.num_steps,
                    f"general/epoch_{player.player_name}": player.num_epochs,
                    f"general/cycle_{player.player_name}": player.num_cycle,
                })

        reward = eval_rollout["next", "reward"].clone()
        del eval_rollout
        self.exploration_coeff = _exploration_coeff

        return reward


class MultiAgentPolicy(BaseTrainableAgent):
    def __init__(
        self,
        action_size: int,
        player_type: int,
        device: torch.device | str,
        run_name: str,
        policy_generator: AgentGenerator,
        embedding_size: int,
        num_defenders: int,
        num_attackers: int,
    ) -> None:
        super().__init__(
            player_type=player_type,
            device=device,
            run_name=run_name,
            agent_id=None,  # MultiAgentPolicy does not have an agent_id
            num_attackers=num_attackers,
            num_defenders=num_defenders,
        )
        self.policy_generator = policy_generator
        self.policies: list[BaseAgent] = [RandomAgent(
            action_size=action_size,
            embedding_size=embedding_size,
            player_type=self.player_type,
            num_defenders=num_defenders,
            num_attackers=num_attackers,
            device=self._device,
            run_name=self.run_name,
            agent_id=0,
        )]

        self._game_to_id_mapper: dict[str, int] = {}

    def save(self) -> None:
        self.policies[-1].save()

    def game_id_to_policy(self, game_id: torch.Tensor) -> BaseAgent:
        assert game_id.ndim == 1, "Game ID must be a 1D tensor."
        policy_id = int.from_bytes(game_id.cpu().numpy().tolist()) % len(self.policies)
        return self.policies[policy_id]

    def load(self, path: str) -> None:
        """
        Load all policies from a given folder.
        """
        assert os.path.isdir(path), f"Path {path} is not a directory."
        for file in os.listdir(path):
            if file.endswith(".pth"):
                policy = self.add_policy()
                policy.load(os.path.join(path, file))

    def add_policy(self) -> BaseAgent:
        policy = self.policy_generator(agent_id=len(self.policies))
        self.policies.append(policy)
        return policy

    def train_cycle(self, tensordict_data: TensorDictBase, replay_buffer: ReplayBuffer, cycle_num: int) -> float:
        assert isinstance(self.policies[-1], BaseTrainableAgent), "Last policy must be a trainable agent."
        return self.policies[-1].train_cycle(tensordict_data, replay_buffer, cycle_num)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # We want to randomize chosen policy only when exploration type is set to RANDOM and we don't currently train it
        if get_interaction_type() == ExplorationType.RANDOM and not self.currently_training:
            policy = self.game_id_to_policy(tensordict["game_id"])
            # policy = self.policies[torch.randint(0, len(self.policies), (1,), dtype=torch.int32).item()]
            return policy.forward(tensordict)
        return self.policies[-1](tensordict)


class MapLogicModuleBase(ABC):
    def __init__(self, env: EnvironmentBase, player_type: int, total_steps: int, device: torch.device | str):
        self._env = env
        self._env_map = env.map
        self._player_type = player_type
        self._total_steps = total_steps
        self._device = device

    @abstractmethod
    def get_action(self, tensordict: TensorDictBase) -> torch.Tensor:
        """
        Get the action based on the current state of the environment.
        """


class FlipItLogicModule(MapLogicModuleBase):
    def get_action(self, tensordict: TensorDictBase) -> torch.Tensor:
        node_owners = tensordict["node_owners_fi"]
        current_step = tensordict["step_count"].item()
        rewards = self._env_map.x[:, 0]
        costs = self._env_map.x[:, 1]

        # Use BeliefState to update belief from node_owners
        belief = belief_state_class(self._env)(Player.attacker, self._env, self._device)
        belief.believed_node_owners = node_owners.clone()
        actions = belief.available_actions()

        # For each reachable node, compute reward for flipping (if not owned)
        best_reward = 0
        best_action = actions[0]
        for action in actions:
            if action < self._env_map.num_nodes:  # Flip action
                if not node_owners[action]:
                    reward = rewards[action].item() * (self._total_steps - current_step) + costs[action].item()
                    if reward > best_reward:
                        best_reward = reward
                        best_action = action
            else:
                # Observe action - for greedy algorithm, any observe is the same
                if best_reward <= 0:
                    best_action = action
                break
        # If no flip is better than observe (which is always 0), best_action_type stays 1
        return torch.tensor([best_action], dtype=torch.int32, device=self._device)


class PoachersLogicModule(MapLogicModuleBase):
    def _distances_to_nearest_reward(self, nodes_collected: torch.Tensor) -> torch.Tensor:
        final_distances = torch.full((self._env_map.num_nodes,), float('inf'), dtype=torch.float32,
                                     device=self._device)

        nodes = torch.where(~nodes_collected & self._env_map.reward_nodes)[0].tolist()
        distances = [0] * len(nodes)
        visited: set[int] = set()

        while nodes:
            current_node = nodes.pop(0)
            distance = distances.pop(0)
            if current_node in visited or distance >= final_distances[current_node].item():
                continue
            visited.add(current_node)
            final_distances[current_node] = distance

            neighbors = self._env_map.get_neighbors(
                torch.tensor([current_node], dtype=torch.int32, device=self._device)
            ).squeeze(0).cpu().tolist()
            for neighbor in neighbors:
                if neighbor != -1 and neighbor not in visited:
                    distances.append(distance + 1)
                    nodes.append(neighbor)

        return final_distances

    def get_action(self, tensordict: TensorDictBase) -> torch.Tensor:
        nodes_collected = tensordict["nodes_collected_fi"]
        nodes_prepared = tensordict["nodes_prepared_fi"]
        position = tensordict["position_seq"][self._player_type, -1].item()

        neighbors = self._env_map.get_neighbors(
            torch.tensor([position], dtype=torch.int32, device=self._device)).squeeze(0)
        valid_neighbors = neighbors[neighbors != -1]
        distances = self._distances_to_nearest_reward(nodes_collected)
        current_distance = distances[position].item()
        if current_distance == 0:
            # Already at not used reward node
            if nodes_prepared[position]:
                action = torch.tensor([6], dtype=torch.int32, device=self._device)  # Collect
            else:
                action = torch.tensor([5], dtype=torch.int32, device=self._device)  # Prepare
        else:
            # Go to the nearest not used reward node
            neighbor_distances = distances[valid_neighbors]
            min_distance = neighbor_distances.min().item()
            distance_indexes = (neighbor_distances == min_distance).nonzero(as_tuple=False).squeeze(-1)
            random_index = distance_indexes[
                torch.randint(0, len(distance_indexes), torch.Size(()), dtype=torch.int32).item()]
            neighbor = valid_neighbors[random_index].item()
            action = (neighbors == neighbor).nonzero(as_tuple=False).reshape(torch.Size((1,)))

        return action


class PoliceLogicModule(MapLogicModuleBase):
    def _distances_to_nearest_reward(self, targets_attacked: torch.Tensor) -> torch.Tensor:
        final_distances = torch.full((self._env_map.num_nodes,), float('inf'), dtype=torch.float32,
                                     device=self._device)

        nodes = torch.where(~targets_attacked & self._env_map.target_nodes)[0].tolist()
        distances = [0] * len(nodes)
        visited: set[int] = set()

        while nodes:
            current_node = nodes.pop(0)
            distance = distances.pop(0)
            if current_node in visited or distance >= final_distances[current_node].item():
                continue
            visited.add(current_node)
            final_distances[current_node] = distance

            neighbors = self._env_map.get_neighbors(
                torch.tensor([current_node], dtype=torch.int32, device=self._device)
            ).squeeze(0).cpu().tolist()
            for neighbor in neighbors:
                if neighbor != -1 and neighbor not in visited:
                    distances.append(distance + 1)
                    nodes.append(neighbor)

        return final_distances

    def get_action(self, tensordict: TensorDictBase) -> torch.Tensor:
        assert self._player_type == 1, "PoliceLogicModule is only implemented for attackers."
        targets_attacked = tensordict["targets_attacked_obs"][..., 1, -1, :].to(torch.bool)
        position = tensordict["position_seq"][self._player_type, -1].item()

        neighbors = self._env_map.get_neighbors(
            torch.tensor([position], dtype=torch.int32, device=self._device)).squeeze(0)
        valid_neighbors = neighbors[neighbors != -1]
        distances = self._distances_to_nearest_reward(targets_attacked)
        current_distance = distances[position].item()
        if current_distance == 0:
            # Already at not used reward node
            action = torch.tensor([4], dtype=torch.int32, device=self._device)  # Attack
        else:
            # Go to the nearest not used reward node
            neighbor_distances = distances[valid_neighbors]
            min_distance = neighbor_distances.min().item()
            distance_indexes = (neighbor_distances == min_distance).nonzero(as_tuple=False).squeeze(-1)
            random_index = distance_indexes[
                torch.randint(0, len(distance_indexes), torch.Size(()), dtype=torch.int32).item()]
            neighbor = valid_neighbors[random_index].item()
            action = (neighbors == neighbor).nonzero(as_tuple=False).reshape(torch.Size((1,)))

        return action


class GreedyOracleAgent(BaseAgent):
    def __init__(
        self,
        action_size: int,
        player_type: int,
        device: torch.device | str,
        run_name: str,
        map_logic: MapLogicModuleBase,
        total_steps: int,
        embedding_size: int,
        num_attackers: int,
        num_defenders: int,
        agent_id: int | None = None,
    ):
        super().__init__(player_type, device, run_name, num_attackers=num_attackers, num_defenders=num_defenders, agent_id=agent_id)
        self._map_logic = map_logic
        self._action_size = action_size
        self._total_steps = total_steps
        self._embedding_size = embedding_size

    def save(self) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = self._map_logic.get_action(tensordict)

        tensordict.update(_tensordict_update_from_action(action, self._embedding_size, self._action_size, self._device))
        return tensordict


class ExplorerAgent(BaseAgent):
    MAX_GAMES = 100

    def __init__(
        self,
        action_size: int,
        player_type: int,
        device: torch.device | str,
        run_name: str,
        total_steps: int,
        embedding_size: int,
        agent_id: int | None = None,
    ):
        super().__init__(player_type, device, run_name, agent_id)
        self._action_size = action_size
        self._total_steps = total_steps
        self.embedding_size = embedding_size

        self.visited_nodes: dict[int, set[int]] = {}
        self.games: list[int] = []

    def save(self) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    @staticmethod
    def get_game_id(game_id: torch.Tensor) -> int:
        return int.from_bytes(game_id.cpu().numpy().tolist())

    def get_generator(self, game_id: torch.Tensor, step_count: int) -> torch.Generator:
        seed_val = (self.get_game_id(game_id) + step_count + 100*self.player_type) % (2**63)
        return torch.Generator().manual_seed(seed_val)

    def add_game(self, game_id: torch.Tensor) -> None:
        game_id_int = self.get_game_id(game_id)
        if game_id_int not in self.games:
            if len(self.games) >= self.MAX_GAMES:
                game_to_remove = self.games.pop(0)
                del self.visited_nodes[game_to_remove]

            self.games.append(game_id_int)
            self.visited_nodes[game_id_int] = set()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone()

        game_id = tensordict["game_id"]
        step_count = tensordict["step_count"].item()
        position = tensordict["position_seq"][self.player_type, -1]
        generator = self.get_generator(game_id, step_count)

        self.add_game(game_id)
        game_id_int = self.get_game_id(game_id)
        self.visited_nodes[game_id_int].add(position.cpu().item())

        reward_info = tensordict["node_reward_info"][Player.attacker.value, -1, :]
        action: torch.Tensor | None = None
        if self.player_type == Player.attacker.value:
            if reward_info[1].item() == 0:
                if reward_info[0].item() == 1:
                    action = torch.tensor(6, dtype=torch.int32, device=self._device)
                else:
                    action = torch.tensor(
                        5 + torch.randint(0, 1, torch.Size(()), dtype=torch.int32, generator=generator).item(),
                        dtype=torch.int32,
                        device=self._device,
                    )

        viable_actions = tensordict["available_moves"][Player.attacker.value, -1, :]
        viable_actions = viable_actions[viable_actions != -1]  # Remove -1 (invalid actions)
        actions_random = torch.randperm(viable_actions.numel(), dtype=torch.int32, generator=generator).to(self._device)
        viable_action = actions_random[~torch.isin(viable_actions, torch.tensor(list(self.visited_nodes[game_id_int]), dtype=torch.int32, device=self._device))]
        if action is None:
            if viable_action.numel() == 0:
                action = torch.tensor(4, dtype=torch.int32, device=self._device)  # Observe
            else:
                action = viable_action[0]

        assert action is not None
        logits = torch.zeros(self._action_size, dtype=torch.float32, device=self._device)
        logits[action] = 1.0
        sample_log_prob = torch.zeros(torch.Size(()), dtype=torch.float32, device=self._device)
        embedding = torch.zeros(self.embedding_size, dtype=torch.float32, device=self._device)
        tensordict.update({
            "action": action,
            "logits": logits,
            "sample_log_prob": sample_log_prob,
            "embedding": embedding,
        })
        return tensordict
