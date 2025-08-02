import inspect
import gc
import sys
from abc import ABC, abstractmethod
from typing import Iterable
from unittest.mock import patch
import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical
from torchrl.data import TensorSpec, Bounded
from torchrl.modules import ActorValueOperator, ValueOperator, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch_geometric.utils import add_self_loops
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer, TensorDictReplayBuffer
import wandb

from config import TrainingConfig, LossConfig, HeadConfig, AgentNNConfig, BackboneConfig
from .base import BaseAgent, BaseTrainableAgent
from .keys_processors import CombinedExtractor, GraphXExtractor, PositionIntLastExtractor, AvailableMovesIntExtractor, LastActionExtractor, StepCountExtractor
from environments.env_mapper import EnvMapper


class BackboneBase(nn.Module, ABC):
    def __init__(self, config: BackboneConfig, extractor: CombinedExtractor, embedding_size: int,
                 max_sequence_size: int, device: torch.device | str) -> None:
        super().__init__()

        self.extractor = extractor
        self.embedding_size = embedding_size
        self.max_sequence_size = max_sequence_size
        self.config = config
        self.device = device

    def to_module(self) -> TensorDictModule:
        return TensorDictModule(
            self, in_keys=self.extractor.in_keys, out_keys=["embedding"]
        )

    @abstractmethod
    def forward(self, *args: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward method that processes the current step or list of steps.
        """


class Backbone(BackboneBase):
    def __init__(self, config: BackboneConfig, extractor: CombinedExtractor, embedding_size: int, max_sequence_size: int, device: torch.device | str) -> None:
        super().__init__(
            config=config,
            extractor=extractor,
            embedding_size=embedding_size,
            max_sequence_size=max_sequence_size,
            device=device,
        )
        assert "x" in self.extractor.input_size and len(self.extractor.input_size) == 1, "Backbone expects a single input size for 'x'."

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.extractor.input_size["x"], self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.embedding_size),
            nn.ReLU(),
        )

    def forward(self, *args: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward method that processes the current step and observed node owners to produce an embedding.
        """
        x = self.extractor.process(*args)["x"]
        out = self.feature_extractor(x.to(self.device))

        return out


class _Linear(nn.Linear):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        weight_initializer: str | None = None,
        bias_initializer: str | None = None,
    ):
        super().__init__(
            in_features=in_channels,
            out_features=out_channels,
            bias=bias,
        )


class GNNBackbone(BackboneBase):
    def __init__(self, config: BackboneConfig, extractor: CombinedExtractor, embedding_size: int,
                 max_sequence_size: int, device: torch.device | str) -> None:
        super().__init__(
            config=config,
            extractor=extractor,
            embedding_size=embedding_size,
            max_sequence_size=max_sequence_size,
            device=device,
        )
        assert "graph_x" in self.extractor.input_size, "Backbone expects 'graph_x' in input size."
        assert "graph_edge_index" in self.extractor.input_size, "Backbone expects 'graph_edge_index' in input keys."
        assert "position" in self.extractor.input_size, "Backbone expects 'position' in input keys."
        assert "available_moves" in self.extractor.input_size, "Backbone expects 'available_moves' in input keys."
        assert "x" in self.extractor.input_size, "Backbone expects 'x' in input keys."

        with patch("torch_geometric.nn.conv.gcn_conv.Linear", side_effect=_Linear):
            from torch_geometric.nn.conv import GCNConv
            self.graph_conv = nn.ModuleList([
                GCNConv(self.extractor.input_size["graph_x"], self.config.hidden_size),
                GCNConv(self.config.hidden_size, self.config.hidden_size),
            ])

        self.output_projection = nn.Sequential(
            nn.Linear(
                self.config.hidden_size * 5 + self.extractor.input_size["x"],
                self.embedding_size,
            ),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
        )

    def forward(self, *args: list[torch.Tensor]) -> torch.Tensor:
        processed = self.extractor.process(*args)
        graph_x = processed["graph_x"]
        graph_edges = processed["graph_edge_index"]
        position = processed["position"]
        available_moves = processed["available_moves"]
        x = processed["x"]

        batch_size: int | None = None
        org_batches: Iterable[int] | None = None
        if graph_x.ndim >= 3:
            org_batches = graph_x.shape[:-2]
            graph_x = graph_x.flatten(0, -3)  # Flatten all but the last two dimensions
            position = position.flatten(0, -2)  # Flatten all but the last dimension
            available_moves = available_moves.flatten(0, -2)  # Flatten all but the last dimension
            if graph_edges.ndim > 2:
                graph_edges = graph_edges.flatten(0, -3)

            batch_size = graph_x.shape[0]
            if graph_edges.ndim == 2:
                graph_edges = graph_edges.repeat(batch_size, 1, 1)  # Repeat edges for each batch
            else:
                graph_edges = graph_edges[0].repeat(batch_size, 1, 1)
            addition = torch.arange(batch_size, device=graph_x.device) * graph_x.shape[2]
            graph_x = graph_x.flatten(0, 1)
            graph_edges += addition.reshape(-1, 1, 1)  # Adjust edges for batch size
            graph_edges = graph_edges.transpose(1, 0).reshape(2, -1)
            all_positions = torch.cat([
                available_moves + addition.reshape(-1, 1),
                position + addition.reshape(-1, 1),
            ], dim=1)
        else:
            assert graph_x.ndim == 2, "Input tensor must be 2D or 3D"
            all_positions = torch.cat([available_moves, position], dim=0)

        graph_x = graph_x.to(self.device)
        graph_edges = graph_edges.to(self.device)
        for conv_layer in self.graph_conv:
            graph_x = conv_layer(graph_x, graph_edges)
            graph_x = torch.relu(graph_x)

        current_x = graph_x[all_positions]
        if batch_size is not None:
            current_x = current_x.reshape(*org_batches, -1)
        else:
            current_x = current_x.reshape(-1)

        current_x = torch.cat([
            current_x,
            x.to(self.device),
        ], dim=-1)

        projected = self.output_projection(current_x)

        return projected


class ObservationEmbedding(BackboneBase):
    def __init__(self, config: BackboneConfig, extractor: CombinedExtractor, embedding_size: int,
                 max_sequence_size: int, device: torch.device | str) -> None:
        super().__init__(
            config=config,
            extractor=extractor,
            embedding_size=embedding_size,
            max_sequence_size=max_sequence_size,
            device=device,
        )
        assert "x" in extractor.input_size, "ObservationEmbedding expects 'x' in input size."

        self.linear_projection = nn.Linear(self.extractor.input_size["x"], self.embedding_size)
        self.activation = nn.ReLU()

    def forward(self, *args: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward method to produce an embedding based on observations.
        """
        x = self.extractor.process(*args)["x"]
        expanded = False
        if x.ndim == 2:
            expanded = True
            x = x.unsqueeze(0)
        x = self.activation(self.linear_projection(x.to(self.device)))
        if expanded:
            x = x.squeeze(0)
        return x


class BackboneTransformer(BackboneBase):
    def __init__(self, config: BackboneConfig, extractor: CombinedExtractor, embedding_size: int,
                 max_sequence_size: int, device: torch.device | str) -> None:
        super().__init__(
            config=config,
            extractor=extractor,
            embedding_size=embedding_size,
            max_sequence_size=max_sequence_size,
            device=device,
        )
        assert self.config.embedding_cls_name is not None, "Embedding class name must be provided in BackboneConfig."

        self.obs_embedding = self._get_obs_embedding_class(self.config.embedding_cls_name)(
            extractor=self.extractor,
            config=self.config,
            embedding_size=self.embedding_size,
            max_sequence_size=self.max_sequence_size,
            device=self.device,
        )
        # Learned Positional Encodings (as per DTQN)
        self.positional_encoder = nn.Embedding(self.max_sequence_size, self.embedding_size)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.config.num_head,
            dim_feedforward=self.config.d_model,
            dropout=self.config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=self.config.num_layers,
            norm=None,  # No final LayerNorm after all N blocks; each block has its own
        )

    @staticmethod
    def _get_obs_embedding_class(embedding_name: str) -> type[nn.Module]:
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if name == embedding_name:
                return cls
        raise ValueError(f"Embedding class '{embedding_name}' not found in {__name__}.")

    @staticmethod
    def _generate_causal_mask(size: int) -> torch.Tensor:
        """Generates a square causal mask for self-attention."""
        mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
        return mask

    def forward(self, *args: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward method that processes the current step and observed node owners to produce an embedding.
        """
        embedded_obs = self.obs_embedding(*args)
        expanded = False
        original_batch_shape = embedded_obs.shape[:-2]  # Keep the original batch shape
        if embedded_obs.ndim == 2:
            expanded = True
            embedded_obs = embedded_obs.unsqueeze(0)
        else:
            embedded_obs = embedded_obs.flatten(0, -3)
        positions_idx = torch.arange(0, self.max_sequence_size, dtype=torch.long, device=embedded_obs.device)
        positions_idx = positions_idx.unsqueeze(0).expand(embedded_obs.shape[0], -1)  # Expand to batch size
        # print(embedded_obs.shape, positions_idx.shape)
        # x = embedded_obs + self.positional_encoder(positions_idx)
        x = embedded_obs + self.positional_encoder(positions_idx)

        causal_mask = self._generate_causal_mask(self.max_sequence_size).to(x.device)

        output_sequence = self.transformer_encoder(x, mask=causal_mask)
        output_sequence = output_sequence[..., -1, :]  # Take the last step's embedding

        if expanded:
            output_sequence = output_sequence.squeeze(0)
        else:
            output_sequence = output_sequence.reshape(*original_batch_shape, -1)

        return output_sequence


class ActorHead(nn.Module):
    def __init__(self, embedding_size: int, player_type: int, action_spec: Bounded, hidden_size: int, device: torch.device | str) -> None:
        super().__init__()

        self.actor_head = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_spec.high+1),
        )
        self.do_sample = True
        self.action_spec = action_spec
        self.player_type = player_type
        self.device = device

    def to_module(self) -> ProbabilisticActor:
        return ProbabilisticActor(
            TensorDictModule(
                self, in_keys=["embedding", "actions_mask"], out_keys=["logits"]
            ),
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )

    def forward(self, x: torch.Tensor, actions_mask: torch.Tensor) -> torch.Tensor:
        x_device = x.device
        is_grad = x.requires_grad
        x = self.actor_head(x.to(self.device))
        x[~actions_mask[..., self.player_type, :]] = -1e8  # Mask invalid actions
        if is_grad:
            return x
        return x.to(x_device)  # Ensure output is on the original device


class ValueHead(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, device: torch.device | str) -> None:
        super().__init__()

        self.device = device
        self.value_head = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def to_module(self) -> ValueOperator:
        return ValueOperator(
            module=self, in_keys=["embedding"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_device = x.device
        is_grad = x.requires_grad
        output = self.value_head(x.to(self.device))
        if is_grad:
            return output
        return output.to(x_device)  # Ensure output is on the original device


class NNAgentPolicy(BaseAgent):
    def __init__(
        self,
        extractor: CombinedExtractor,
        max_sequence_size: int,
        action_size: int,
        player_type: int,
        head_config: HeadConfig,
        backbone_config: BackboneConfig,
        agent_config: AgentNNConfig,
        device: torch.device | str,
        run_name: str,
        agent_id: int | None = None,
    ) -> None:
        super().__init__(
            player_type=player_type,
            device=device,
            run_name=run_name,
            agent_id=agent_id,
        )
        self.action_size = action_size

        action_spec = Bounded(
            shape=torch.Size((1,)),
            low=0,
            high=action_size - 1,
            dtype=torch.int32,
        )

        backbone = self._get_backbone_class(backbone_config.cls_name)(
            config=backbone_config,
            extractor=extractor,
            embedding_size=agent_config.embedding_size,
            max_sequence_size=max_sequence_size,
            device=self._device,
        ).to(self._device)

        actor_head = ActorHead(
            embedding_size=agent_config.embedding_size,
            player_type=player_type,
            action_spec=action_spec,
            hidden_size=head_config.hidden_size,
            device=self._device,
        ).to(self._device)
        value_head = ValueHead(
            embedding_size=agent_config.embedding_size,
            hidden_size=head_config.hidden_size,
            device=self._device,
        ).to(self._device)

        self.agent = ActorValueOperator(
            backbone.to_module(),
            actor_head.to_module(),
            value_head.to_module(),
        )

    @staticmethod
    def _get_backbone_class(cls_name: str) -> type[BackboneBase]:
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
            if name == cls_name:
                assert issubclass(cls, BackboneBase), f"Class '{cls_name}' is not a subclass of BackboneBase."
                return cls
        raise ValueError(f"Backbone class '{cls_name}' not found in {__name__}.")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        self.agent = self.agent.to(self._device)
        return self.agent.get_policy_operator()(tensordict.to(self._device))


class TrainableNNAgentPolicy(NNAgentPolicy, BaseTrainableAgent):
    def __init__(
        self,
        player_type: int,
        max_sequence_size: int,
        extractor: CombinedExtractor,
        action_size: int,
        device: torch.device | str,
        run_name: str,
        training_config: TrainingConfig,
        loss_config: LossConfig,
        head_config: HeadConfig,
        backbone_config: BackboneConfig,
        agent_config: AgentNNConfig,
        env_type: EnvMapper,
        agent_id: int | None = None,
        scheduler_steps: int | None = None,
        add_logs: bool = True,
    ) -> None:
        super().__init__(
            extractor=extractor,
            max_sequence_size=max_sequence_size,
            action_size=action_size,
            player_type=player_type,
            head_config=head_config,
            backbone_config=backbone_config,
            agent_config=agent_config,
            device=device,
            run_name=run_name,
            agent_id=agent_id,
        )

        self._training_config = training_config
        self._loss_config = loss_config
        self._max_grad_norm = loss_config.max_grad_norm
        self._add_logs = add_logs
        self._env_type = env_type

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
            gamma=loss_config.gamma, lmbda=loss_config.lmbda, value_network=self.agent.get_value_head(), average_gae=False#, device=torch.device("cuda:0")
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
        loss_vals = self.loss(tensordict)
        loss_value = (
            loss_vals["loss_objective"]  # **2
            + loss_vals["loss_critic"]  # **2
            + loss_vals["loss_entropy"]  # **2
        )

        self.optimizer.zero_grad()
        loss_value.backward()

        torch.nn.utils.clip_grad_norm_(self.loss.parameters(), self._loss_config.max_grad_norm)
        self.optimizer.step()

        self.num_steps += tensordict.batch_size.numel()  # add number of steps in environment

        if self._add_logs:
            wandb.log({
                f"loss/objective_step_{self.player_name}": loss_vals["loss_objective"].detach().cpu().item(),
                f"loss/critic_step_{self.player_name}": loss_vals["loss_critic"].detach().cpu().item(),
                f"loss/entropy_step_{self.player_name}": loss_vals["loss_entropy"].detach().cpu().item(),
                f"train/entropy_step_{self.player_name}": loss_vals["entropy"].detach().cpu().item(),
                f"train/state_value_step_{self.player_name}": tensordict["state_value"].detach().cpu().mean().item(),
                f"train/advantage_step_{self.player_name}": tensordict["advantage"].detach().cpu().mean().item(),
                f"general/step_{self.player_name}": self.num_steps,
                f"general/epoch_{self.player_name}": self.num_epochs,
                f"general/cycle_{self.player_name}": self.num_cycle,
            })

        return loss_vals.detach().cpu()

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
            subdata, info = replay_buffer.sample(self._training_config.sub_batch_size, return_info=True)
            # The ClipPPOLoss will normalize advantage internally. We apply weights
            # before this normalization happens by modifying the tensordict.
            # subdata["advantage"] = subdata["advantage"] * info["_weight"].unsqueeze(-1)
            loss_vals = self._training_step(subdata.to(self._device))

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

    def train_cycle(self, tensordict_data: TensorDictBase, replay_buffer: TensorDictReplayBuffer, cycle_num: int) -> float:
        self.train()
        tensordict_data = tensordict_data.flatten()

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
        # tensordict_data = tensordict_data.to(self._device)

        cycle_data = {
            "loss_objective": [],
            "loss_critic": [],
            "loss_entropy": [],
            "state_value": [],
            "entropy": [],
            "advantage": [],
        }

        for _ in range(self._training_config.epochs_per_batch):
            with torch.no_grad():
                tensordict_data["next"].update(self.agent.module[0](tensordict_data["next"]))
                tensordict_data.update(self.agent.module[0](tensordict_data))
            self.advantage(tensordict_data)
            #priorities = tensordict_data["advantage"].abs() + 1e-6  # Add epsilon for stability
            # The sampler expects a "_priority" key. Squeeze to remove the last dim.
            #tensordict_data.set("priority", priorities.squeeze(-1))
            #tensordict_data.set("index", torch.arange(tensordict_data.batch_size.numel(), device=tensordict_data.device).reshape(*tensordict_data.batch_size))

            replay_buffer.empty()
            replay_buffer.extend(tensordict_data)
            replay_buffer.update_tensordict_priority(tensordict_data)

            epoch_data = self._train_epoch(replay_buffer, cycle_num)

            for key in cycle_data:
                cycle_data[key].extend(epoch_data[key])

        reward = tensordict_data["next"]["reward"].mean().item()
        if self._add_logs:
            log_dict = {
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
            }
            if self._env_type == EnvMapper.flipit:
                total_actions = tensordict_data["action"].numel()
                num_nodes = self.action_size // 2
                flip_actions = (tensordict_data["action"] // num_nodes == 0).sum().item() / total_actions
                observe_actions = (tensordict_data["action"] // num_nodes == 1).sum().item() / total_actions
                log_dict.update({
                    f"actions/flip_{self.player_name}": flip_actions,
                    f"actions/observe_{self.player_name}": observe_actions,
                })
            else:
                total_actions = tensordict_data["action"].numel()
                for i in range(self.action_size):
                    action_count = (tensordict_data["action"] == i).sum().item() / total_actions
                    log_dict[f"actions/action_{i}_{self.player_name}"] = action_count

                log_dict["actions/avg_length"] = tensordict_data["next"]["done"].numel() / max(tensordict_data["next"]["done"].sum().item(), 1)

            wandb.log(log_dict)

        del tensordict_data
        #torch.cuda.empty_cache()
        #gc.collect()

        self.num_cycle += 1
        if self.scheduler is not None:
            self.scheduler.step()

        return reward

