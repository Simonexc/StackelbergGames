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
from environments.flipit_utils import BeliefState
from environments.flipit_geometric import FlipItMap


class RandomAgent(BaseAgent):
    def __init__(
        self,
        action_size: int,
        embedding_size: int,
        player_type: int,
        device: torch.device | str,
        run_name: str,
        agent_id: int | None = None,
    ) -> None:
        super().__init__(player_type, device, run_name, agent_id)
        self.embedding_size = embedding_size
        self.action_size = action_size

    def save(self) -> None:
        # RandomAgent does not save its state
        pass

    def load(self, path: str) -> None:
        # RandomAgent does not load any state
        pass

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = torch.randint(0, self.action_size, torch.Size(()), dtype=torch.int32, device=self._device)
        logits = torch.zeros(self.action_size, dtype=torch.float32, device=self._device)
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
        with torch.no_grad():
            defender_output = self._defender_module(tensordict.clone())
            attacker_output = self._attacker_module(tensordict)

        attacker_output.update({
            "action": torch.stack([defender_output["action"].clone(), attacker_output["action"]], dim=-1),
            "logits": torch.stack([defender_output["logits"].clone(), attacker_output["logits"]], dim=-1),
            "sample_log_prob": torch.stack([defender_output["sample_log_prob"].clone(), attacker_output["sample_log_prob"]], dim=-1),
            "embedding": torch.stack([defender_output["embedding"].clone(), attacker_output["embedding"]], dim=-1),
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
    ) -> None:
        super().__init__(
            player_type=player_type,
            device=device,
            run_name=run_name,
            agent_id=None,  # MultiAgentPolicy does not have an agent_id
        )
        self.policy_generator = policy_generator
        self.policies: list[BaseAgent] = [RandomAgent(
            action_size=action_size,
            embedding_size=embedding_size,
            player_type=self.player_type,
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


class GreedyOracleAgent(BaseAgent):
    def __init__(
        self,
        action_size: int,
        player_type: int,
        device: torch.device | str,
        run_name: str,
        env_map: FlipItMap,
        total_steps: int,
        embedding_size: int,
        agent_id: int | None = None,
    ):
        super().__init__(player_type, device, run_name, agent_id)
        self._env_map = env_map
        self._action_size = action_size
        self._total_steps = total_steps
        self._rewards = env_map.x[:, 0]
        self._costs = env_map.x[:, 1]
        self.embedding_size = embedding_size

    def save(self) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        node_owners = tensordict["node_owners"]
        current_step = tensordict["step_count"].item()

        # Use BeliefState to update belief from node_owners
        belief = BeliefState(Player.attacker, type('DummyEnv', (), {'map': self._env_map})(), self._device)
        belief.believed_node_owners = node_owners.clone()
        reachable_nodes = belief.nodes_reachable()
        if not reachable_nodes:
            # fallback: pick any node
            reachable_nodes = list(range(self.num_nodes))

        # For each reachable node, compute reward for flipping (if not owned)
        best_reward = 0
        best_action_type = 1  # default to observe
        best_target_node = reachable_nodes[0]
        for node in reachable_nodes:
            if not node_owners[node]:
                reward = self._rewards[node].item() * (self._total_steps - current_step) + self._costs[node].item()
                if reward > best_reward:
                    best_reward = reward
                    best_action_type = 0  # flip
                    best_target_node = node
        # If no flip is better than observe (which is always 0), best_action_type stays 1
        action = torch.tensor(best_action_type * len(node_owners) + best_target_node, dtype=torch.int32, device=self._device)
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
