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
from environments.config import EnvMapper


class Backbone(nn.Module):
    def __init__(self, extractor: CombinedExtractor, embedding_size: int, device: torch.device | str, hidden_size: int) -> None:
        super().__init__()

        self.extractor = extractor
        self.embedding_size = embedding_size
        self.device = device

        self.feature_extractor = nn.Sequential(
            nn.Linear(extractor.input_size, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size, device=device),
            nn.ReLU(),
        )

    def to_module(self) -> TensorDictModule:
        return TensorDictModule(
            self, in_keys=self.extractor.in_keys, out_keys=["embedding"]
        )

    def forward(self, *args: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward method that processes the current step and observed node owners to produce an embedding.
        """
        x = self.extractor.process(*args)
        out = self.feature_extractor(x)

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


class GNNBackbone(nn.Module):
    def __init__(
        self,
        extractor: CombinedExtractor,
        embedding_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()

        self.graph_x_extractor = GraphXExtractor(extractor._player_type, extractor._env)
        self.position_extractor = PositionIntLastExtractor(extractor._player_type, extractor._env)
        self.available_moves_extractor = AvailableMovesIntExtractor(extractor._player_type, extractor._env)
        self.last_action_extractor = LastActionExtractor(extractor._player_type, extractor._env)
        self.step_count_extractor = StepCountExtractor(extractor._player_type, extractor._env)
        with patch("torch_geometric.nn.conv.gcn_conv.Linear", side_effect=_Linear):
            from torch_geometric.nn.conv import GCNConv
            self.graph_conv = nn.ModuleList([
                GCNConv(5, hidden_size),
                GCNConv(hidden_size, hidden_size),
            ])
        self.output_projection = nn.Sequential(
            nn.Linear(
                hidden_size*5
                + self.last_action_extractor.expected_size
                + self.step_count_extractor.expected_size,
                embedding_size,
            ),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
        )

    def to_module(self) -> TensorDictModule:
        return TensorDictModule(
            self, in_keys=[self.graph_x_extractor.KEY, "graph_edge_index", self.position_extractor.KEY, self.available_moves_extractor.KEY, self.last_action_extractor.KEY, self.step_count_extractor.KEY], out_keys=["embedding"]
        )

    def forward(self, graph_x: torch.Tensor, graph_edges: torch.Tensor, position_seq: torch.Tensor, available_moves: torch.Tensor, actions: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        x = self.graph_x_extractor.process(graph_x).clone()
        position = self.position_extractor.process(position_seq).clone()
        graph_edges = graph_edges.clone()
        available_moves = self.available_moves_extractor.process(available_moves)
        last_action = self.last_action_extractor.process(actions)
        steps = self.step_count_extractor.process(steps.clone())

        batch_size: int | None = None
        if x.ndim == 3:
            batch_size = x.shape[0]
            addition = torch.arange(batch_size, device=x.device) * x.shape[1]
            x = x.reshape(-1, *x.shape[2:])
            graph_edges += addition.reshape(-1, 1, 1)  # Adjust edges for batch size
            graph_edges = graph_edges.transpose(1, 0).reshape(2, -1)
            position += addition
            all_positions = torch.cat([available_moves + addition.reshape(-1, 1), position.unsqueeze(1)], dim=1)
        else:
            assert x.ndim == 2, "Input tensor must be 2D or 3D"
            all_positions = torch.cat([available_moves, position.unsqueeze(0)], dim=0)

        for conv_layer in self.graph_conv:
            x = conv_layer(x, graph_edges)
            x = torch.relu(x)

        current_x = x[all_positions]
        if batch_size is not None:
            current_x = current_x.reshape(batch_size, -1)
        else:
            current_x = current_x.reshape(-1)

        current_x = torch.cat([
            current_x,
            last_action,
            steps,
        ], dim=-1)

        projected = self.output_projection(current_x)
        if batch_size is not None:
            projected = projected.reshape(batch_size, -1)

        return projected


class ObservationEmbedding(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, device: torch.device | str) -> None:
        super().__init__()

        self.linear_projection = nn.Linear(input_size, embedding_size, device=device)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method to produce an embedding based on observations.
        """
        return self.activation(self.linear_projection(x))


class BackboneTransformer(nn.Module):
    def __init__(
        self,
        extractor: CombinedExtractor,
        max_sequence_size: int,
        embedding_size: int,
        player_type: int,
        device: torch.device | str,
        d_model: int,
        num_head: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.extractor = extractor
        self.player_type = player_type
        self.embedding_size = embedding_size
        self.device = device
        self.max_sequence_size = max_sequence_size

        self.d_model = d_model

        self.obs_embedding = ObservationEmbedding(
            input_size=self.extractor.input_size,
            embedding_size=self.embedding_size,
            device=device,
        )
        # Learned Positional Encodings (as per DTQN)
        self.positional_encoder = nn.Embedding(self.max_sequence_size, self.embedding_size, device=self.device)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=num_head,
            dim_feedforward=d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
            device=device,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=num_layers,
            norm=None,  # No final LayerNorm after all N blocks; each block has its own
        )

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generates a square causal mask for self-attention."""
        mask = torch.triu(torch.full((size, size), float('-inf'), device=self.device), diagonal=1)
        return mask

    def to_module(self) -> TensorDictModule:
        return TensorDictModule(
            self, in_keys=self.extractor.in_keys, out_keys=["embedding"]
        )

    def forward(self, *args: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward method that processes the current step and observed node owners to produce an embedding.
        """
        x = self.extractor.process(*args)

        expanded = False
        if x.ndim == 2:
            expanded = True
            x = x.unsqueeze(0)
        embedded_obs = self.obs_embedding(x)
        positions_idx = torch.arange(0, self.max_sequence_size, dtype=torch.long, device=self.device)
        positions_idx = positions_idx.unsqueeze(0).expand(embedded_obs.size(0), -1)  # Expand to batch size

        x = embedded_obs + self.positional_encoder(positions_idx)

        causal_mask = self._generate_causal_mask(self.max_sequence_size)

        output_sequence = self.transformer_encoder(x, mask=causal_mask)
        output_sequence = output_sequence[..., -1, :]  # Take the last step's embedding

        if expanded:
            output_sequence = output_sequence.squeeze(0)
        return output_sequence


class ActorHead(nn.Module):
    def __init__(self, embedding_size: int, player_type: int, device: torch.device | str, action_spec: Bounded, hidden_size: int) -> None:
        super().__init__()

        self.actor_head = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, action_spec.high+1, device=device),
        )
        self.do_sample = True
        self.action_spec = action_spec
        self.player_type = player_type

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
        x = self.actor_head(x).clone()
        x[~actions_mask[..., self.player_type, :]] = -1e8  # Mask invalid actions
        return x


class ValueHead(nn.Module):
    def __init__(self, embedding_size: int, device: torch.device | str, hidden_size: int) -> None:
        super().__init__()

        self.device = device
        self.value_head = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, device=device),
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

        if backbone_config.use_transformer:
            backbone = BackboneTransformer(
                extractor=extractor,
                max_sequence_size=max_sequence_size,
                embedding_size=agent_config.embedding_size,
                player_type=player_type,
                device=self._device,
                d_model=backbone_config.d_model,
                num_head=backbone_config.num_head,
                num_layers=backbone_config.num_layers,
                dropout=backbone_config.dropout,
            )
        else:
            # backbone = GNNBackbone(
            #     extractor=extractor,
            #     embedding_size=agent_config.embedding_size,
            #     hidden_size=backbone_config.hidden_size,
            # ).to(self._device)
            #backbone.load_state_dict(torch.load("test.pth", map_location=self._device))
            # freeze weights
            #for param in backbone.parameters():
            #    param.requires_grad = False

            backbone = Backbone(
                extractor=extractor,
                embedding_size=agent_config.embedding_size,
                device=self._device,
                hidden_size=backbone_config.hidden_size,
            )

        actor_head = ActorHead(
            embedding_size=agent_config.embedding_size,
            player_type=player_type,
            device=self._device,
            action_spec=action_spec,
            hidden_size=head_config.hidden_size,
        )
        value_head = ValueHead(
            embedding_size=agent_config.embedding_size,
            device=self._device,
            hidden_size=head_config.hidden_size,
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
            gamma=loss_config.gamma, lmbda=loss_config.lmbda, value_network=self.agent.get_value_head(), average_gae=False
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
            subdata, info = replay_buffer.sample(self._training_config.sub_batch_size, return_info=True)
            # The ClipPPOLoss will normalize advantage internally. We apply weights
            # before this normalization happens by modifying the tensordict.
            # subdata["advantage"] = subdata["advantage"] * info["_weight"].unsqueeze(-1)

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

    def train_cycle(self, tensordict_data: TensorDictBase, replay_buffer: TensorDictReplayBuffer, cycle_num: int) -> float:
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
            with torch.no_grad():
                tensordict_data["next"].update(self.agent.module[0](tensordict_data["next"]))
                tensordict_data.update(self.agent.module[0](tensordict_data))
            self.advantage(tensordict_data)
            priorities = tensordict_data["advantage"].abs() + 1e-6  # Add epsilon for stability
            # The sampler expects a "_priority" key. Squeeze to remove the last dim.
            tensordict_data.set("priority", priorities.squeeze(-1))
            tensordict_data.set("index", torch.arange(tensordict_data.batch_size.numel(), device=self._device))

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
                log_dict["actions/avg_length"] = tensordict_data["next"]["done"].numel() / tensordict_data["next"]["done"].sum().item()

            wandb.log(log_dict)

        self.num_cycle += 1
        if self.scheduler is not None:
            self.scheduler.step()

        return reward

