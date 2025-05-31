import torch
import wandb
from tensordict.nn import set_interaction_type as set_exploration_type, InteractionType as ExplorationType
from tensordict.nn.probabilistic import interaction_type as get_interaction_type
from torch import nn
from tensordict import TensorDictBase
from torchrl.data import ReplayBuffer
from torchrl.envs import EnvBase

from .base import BaseAgent, BaseTrainableAgent
from .generator import AgentGenerator


class RandomAgent(BaseAgent):
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
    def __init__(self, defender_module: nn.Module, attacker_module: nn.Module) -> None:
        super().__init__()
        self.defender_module = defender_module
        self.attacker_module = attacker_module

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        defender_output = self.defender_module(tensordict.clone())
        attacker_output = self.attacker_module(tensordict)

        attacker_output.update({
            "action": torch.stack([defender_output["action"], attacker_output["action"]], dim=-1),
            "logits": torch.stack([defender_output["logits"], attacker_output["logits"]], dim=-1),
            "sample_log_prob": torch.stack([defender_output["sample_log_prob"], attacker_output["sample_log_prob"]], dim=-1),
            "embedding": torch.stack([defender_output["embedding"], attacker_output["embedding"]], dim=-1),
        })
        return attacker_output

    def evaluate(self, env: EnvBase, rollout_num: int, current_player: int, add_logs: bool = True) -> torch.Tensor:
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

        return reward

    def single_run(self, env: EnvBase, current_player: int, add_logs: bool = True, exploration_type = ExplorationType.DETERMINISTIC) -> torch.Tensor:
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

        return reward


class MultiAgentPolicy(BaseTrainableAgent):
    def __init__(
        self,
        action_size: int,
        player_type: int,
        embedding_size: int,
        device: torch.device | str,
        run_name: str,
        policy_generator: AgentGenerator,
    ) -> None:
        super().__init__(
            action_size=action_size,
            player_type=player_type,
            embedding_size=embedding_size,
            device=device,
            run_name=run_name,
            agent_id=None,  # MultiAgentPolicy does not have an agent_id
        )
        self.policy_generator = policy_generator
        self.policies: list[BaseAgent] = [RandomAgent(
            action_size=self.action_size,
            player_type=self.player_type,
            device=self._device,
            embedding_size=self.embedding_size,
            run_name=self.run_name,
            agent_id=0,
        )]

    def save(self) -> None:
        self.policies[-1].save()

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
            policy = self.policies[torch.randint(0, len(self.policies), (1,), dtype=torch.int32).item()]
            return policy.forward(tensordict)
        return self.policies[-1](tensordict)
