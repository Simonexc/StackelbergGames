import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torchrl.data import TensorSpec, Bounded
from torchrl.modules import ActorValueOperator, ValueOperator, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
import wandb

from config import TrainingConfig, LossConfig
from .base import BaseAgent, BaseTrainableAgent


class Backbone(nn.Module):
    def __init__(self, num_nodes: int, embedding_size: int, player_type: int, device: torch.device | str) -> None:
        super().__init__()

        self.player_type = player_type
        self.num_nodes = num_nodes
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.num_nodes + 1, 32, device=device),
            nn.ReLU(),
            nn.Linear(32, embedding_size, device=device),
            nn.ReLU(),
        )

    def to_module(self) -> TensorDictModule:
        return TensorDictModule(
            self, in_keys=["step_count", "observed_node_owners"], out_keys=["embedding"]
        )

    def forward(self, current_step: torch.Tensor, observed_node_owners: torch.Tensor) -> torch.Tensor:
        # we need to expand the current step to match the batch size of the observed node owners
        current_step = current_step / self.num_nodes - 0.5

        observed_node_owners = observed_node_owners[..., self.player_type, :] / 2
        # we need to concatenate the two tensors
        x = torch.cat([current_step, observed_node_owners], dim=-1)
        return self.feature_extractor(x)


class ActorHead(nn.Module):
    def __init__(self, num_nodes: int, embedding_size: int, device: torch.device | str, action_spec: TensorSpec) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.actor_head = nn.Sequential(
            nn.Linear(embedding_size, num_nodes, device=device),
            #nn.ReLU(),
            #nn.Linear(256, num_nodes, device=device),
        )
        self.do_sample = True
        self.action_spec = action_spec

    def to_module(self) -> ProbabilisticActor:
        return ProbabilisticActor(
            TensorDictModule(
                self, in_keys=["embedding"], out_keys=["logits"]
            ),
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:#tuple[torch.Tensor, torch.Tensor]:
        x = self.actor_head(x)

        # dist = Categorical(logits=x)
        # if self.do_sample:
        #     action_id = dist.sample()
        # else:
        #     action_id = dist.probs.argmax(dim=-1).to(torch.int32)
        #
        # action = action_id // self.num_nodes
        # action_target = action_id % self.num_nodes
        return x
        # return (
        #     torch.stack([action, action_target], dim=-1),
        #     x,
        # )


class ValueHead(nn.Module):
    def __init__(self, embedding_size: int, device: torch.device | str) -> None:
        super().__init__()

        self.device = device
        self.value_head = nn.Sequential(
            nn.Linear(embedding_size, 1, device=device),
            #nn.ReLU(),
            #nn.Linear(256, 1, device=device),
        )

    def to_module(self) -> ValueOperator:
        return ValueOperator(
            module=self, in_keys=["embedding"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(x)


class NNAgentPolicy(BaseAgent):
    def __init__(
        self,
        num_nodes: int,
        player_type: int,
        embedding_size: int,
        device: torch.device | str,
        run_name: str,
        agent_id: int | None = None,
    ) -> None:
        super().__init__(
            action_size=num_nodes,
            player_type=player_type,
            embedding_size=embedding_size,
            device=device,
            run_name=run_name,
            agent_id=agent_id,
        )

        action_spec = Bounded(
            shape=torch.Size((1,)),
            low=0,
            high=self.action_size - 1,
            dtype=torch.int32,
        )

        backbone = Backbone(
            num_nodes=self.action_size,
            embedding_size=self.embedding_size,
            player_type=player_type,
            device=self._device,
        )
        actor_head = ActorHead(
            num_nodes=self.action_size,
            embedding_size=self.embedding_size,
            device=self._device,
            action_spec=action_spec,
        )
        value_head = ValueHead(
            embedding_size=self.embedding_size,
            device=self._device,
        )

        self.agent = ActorValueOperator(
            backbone.to_module(),
            actor_head.to_module(),
            value_head.to_module(),
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.agent.get_policy_operator()(tensordict)


class TrainableNNAgentPolicy(NNAgentPolicy, BaseTrainableAgent):
    def __init__(
        self,
        num_nodes: int,
        player_type: int,
        embedding_size: int,
        device: torch.device | str,
        run_name: str,
        training_config: TrainingConfig,
        loss_config: LossConfig,
        agent_id: int | None = None,
        scheduler_steps: int | None = None,
        add_logs: bool = True,
    ) -> None:
        super().__init__(
            num_nodes=num_nodes,
            player_type=player_type,
            embedding_size=embedding_size,
            device=device,
            run_name=run_name,
            agent_id=agent_id,
        )

        self._training_config = training_config
        self._loss_config = loss_config
        self._max_grad_norm = loss_config.max_grad_norm
        self._add_logs = add_logs

        self.loss = ClipPPOLoss(
            actor_network=self.agent.get_policy_operator(),
            critic_network=self.agent.get_value_head(),
            clip_epsilon=loss_config.clip_epsilon,
            entropy_bonus=bool(loss_config.entropy_eps),
            entropy_coef=loss_config.entropy_eps,
            normalize_advantage=True,
            critic_coef=loss_config.critic_coef,
            loss_critic_type="smooth_l1",
        )

        self.advantage = GAE(
            gamma=loss_config.gamma, lmbda=loss_config.lmbda, value_network=self.agent.get_value_operator()
        )

        self.optimizer = torch.optim.Adam(
            self.loss.parameters(), lr=loss_config.learning_rate
        )

        if scheduler_steps is None:
            self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=scheduler_steps,
                eta_min=1e-9,
            )

    def _training_step(self, tensordict: TensorDictBase) -> dict:
        loss_vals = self.loss(tensordict.to(self._device))
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        self.optimizer.zero_grad()
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(self.loss.parameters(), self._loss_config.max_grad_norm)
        self.optimizer.step()

        self.num_steps += tensordict.batch_size.numel()  # add number of steps in environment

        if self._add_logs:
            wandb.log({
                f"loss/objective_step_{self.player_name}": loss_vals["loss_objective"].item(),
                f"loss/critic_step_{self.player_name}": loss_vals["loss_critic"].item(),
                f"loss/entropy_step_{self.player_name}": loss_vals["loss_entropy"].item(),
                f"train/entropy_step_{self.player_name}": loss_vals["entropy"].item(),
                f"train/state_value_step_{self.player_name}": tensordict["state_value"].mean().item(),
                f"train/advantage_step_{self.player_name}": tensordict["advantage"].mean().item(),
                f"general/step_{self.player_name}": self.num_steps,
                f"general/epoch_{self.player_name}": self.num_epochs,
                f"general/cycle_{self.player_name}": self.num_cycle,
            })

        return loss_vals

    def _train_epoch(self, replay_buffer: ReplayBuffer, cycle_num: int) -> dict:
        epoch_data = {
            "loss_objective": [],
            "loss_critic": [],
            "loss_entropy": [],
            "state_value": [],
            "entropy": [],
            "advantage": [],
        }
        for current_step in range(self._training_config.steps_per_batch // self._training_config.sub_batch_size):
            subdata = replay_buffer.sample(self._training_config.sub_batch_size)
            loss_vals = self._training_step(subdata)

            epoch_data["loss_objective"].append(loss_vals["loss_objective"].item())
            epoch_data["loss_critic"].append(loss_vals["loss_critic"].item())
            epoch_data["loss_entropy"].append(loss_vals["loss_entropy"].item())
            epoch_data["state_value"].append(subdata["state_value"].mean().item())
            epoch_data["entropy"].append(loss_vals["entropy"].item())
            epoch_data["advantage"].append(subdata["advantage"].mean().item())

        if self._add_logs:
            wandb.log({
                f"loss/objective_epoch_{self.player_name}": np.array(epoch_data["loss_objective"]).mean(),
                f"loss/critic_epoch_{self.player_name}": np.array(epoch_data["loss_critic"]).mean(),
                f"loss/entropy_epoch_{self.player_name}": np.array(epoch_data["loss_entropy"]).mean(),
                f"train/state_value_epoch_{self.player_name}": np.array(epoch_data["state_value"]).mean(),
                f"train/entropy_epoch_{self.player_name}": np.array(epoch_data["entropy"]).mean(),
                f"train/advantage_epoch_{self.player_name}": np.array(epoch_data["advantage"]).mean(),
                f"general/step_{self.player_name}": self.num_steps,
                f"general/epoch_{self.player_name}": self.num_epochs,
                f"general/cycle_{self.player_name}": self.num_cycle,
            })

        self.num_epochs += 1

        return epoch_data

    def train_cycle(self, tensordict_data: TensorDictBase, replay_buffer: ReplayBuffer, cycle_num: int) -> float:
        self.train()

        # update tensordict
        tensordict_data.update({
            "action": tensordict_data["action"][..., self.player_type],
            "logits": tensordict_data["logits"][..., self.player_type],
            "sample_log_prob": tensordict_data["sample_log_prob"][..., self.player_type],
            "embedding": tensordict_data["embedding"][..., self.player_type],
        })
        tensordict_data["next"].update({
            "reward": tensordict_data["next"]["reward"][..., self.player_type].unsqueeze(-1),  # retain dimensionality
        })

        cycle_data = {
            "loss_objective": [],
            "loss_critic": [],
            "loss_entropy": [],
            "state_value": [],
            "entropy": [],
            "advantage": [],
        }

        for _ in range(self._training_config.epochs_per_batch):
            self.advantage(tensordict_data)
            replay_buffer.extend(tensordict_data)

            epoch_data = self._train_epoch(replay_buffer, cycle_num)

            for key in cycle_data:
                cycle_data[key].extend(epoch_data[key])

        reward = tensordict_data["next"]["reward"].mean().item()
        if self._add_logs:
            wandb.log({
                f"loss/objective_{self.player_name}": np.array(cycle_data["loss_objective"]).mean(),
                f"loss/critic_{self.player_name}": np.array(cycle_data["loss_critic"]).mean(),
                f"loss/entropy_{self.player_name}": np.array(cycle_data["loss_entropy"]).mean(),
                f"train/state_value_{self.player_name}": np.array(cycle_data["state_value"]).mean(),
                f"train/entropy_{self.player_name}": np.array(cycle_data["entropy"]).mean(),
                f"train/advantage_{self.player_name}": np.array(cycle_data["advantage"]).mean(),
                f"train/reward_mean_{self.player_name}": reward,
                f"train/reward_std_{self.player_name}": tensordict_data["next", "reward"].std().item(),
                f"train/lr_attacker": self.optimizer.param_groups[0]["lr"],
                f"general/step_{self.player_name}": self.num_steps,
                f"general/epoch_{self.player_name}": self.num_epochs,
                f"general/cycle_{self.player_name}": self.num_cycle,
            })

        self.num_cycle += 1

        return reward


