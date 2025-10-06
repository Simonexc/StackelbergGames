from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final
import uuid

import torch
from tensordict.base import TensorDictBase
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Bounded, TensorSpec, Composite, Unbounded

from .base_map import EnvMapBase

if TYPE_CHECKING:
    from config import EnvConfig


class EnvironmentBase(EnvBase, ABC):
    def __init__(
        self,
        config: "EnvConfig",
        env_map: EnvMapBase,
        device: torch.device | str | None = None,
        batch_size: torch.Size | None = None,
        num_defenders: int = 1,
        num_attackers: int = 1,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if batch_size is None:
            batch_size = torch.Size([])

        assert batch_size == torch.Size([]), "Batch size must be a empty."

        self.map = env_map.to(device)
        self.num_steps = config.num_steps
        self.config = config
        self.num_defenders = num_defenders
        self.num_attackers = num_attackers
        self._generator = torch.Generator(device=device)
        self._generator.seed()

        super().__init__(device=device, batch_size=batch_size)
        self._make_spec()
        assert isinstance(self.action_spec, Bounded), "Action shape should be of type Bounded."

        # Initialize game_id tensor placeholder; actual UUIDs will be set in _reset
        self.game_id = torch.empty((*self.batch_size, 16), dtype=torch.uint8, device=self.device)
        self.step_count = torch.zeros((*self.batch_size, 1), dtype=torch.int32, device=self.device)

    def create_from_self(self) -> "EnvironmentBase":
        """
        Create a new instance of the environment with the same configuration.
        This method is used to create a new environment instance with the same settings.
        """
        return type(self)(self.config, self.map, device=self.device, batch_size=self.batch_size)

    @property
    @abstractmethod
    def action_size(self) -> int:
        """
        Define the action size for the environment.
        This method should be implemented by subclasses to specify the action space size.
        """

    @property
    @abstractmethod
    def graph_x_size(self) -> int:
        """
        Define the size of the node features tensor `x` in the environment's graph.
        This method should be implemented by subclasses to specify the feature size.
        """

    @property
    @abstractmethod
    def actions_mask(self) -> torch.Tensor:
        """
        Define the action mask for the given state of the environment.
        This method should be implemented by subclasses to specify which actions are valid.
        """

    @abstractmethod
    def _get_observation_spec(self) -> dict[str, TensorSpec]:
        """
        Define the observation spec for the environment.
        This method should be implemented by subclasses to specify the observation space.
        """

    @final
    def _set_seed(self, seed: int | None) -> None:
        #self._generator.manual_seed(seed)
        pass

    @final
    def _make_spec(self) -> None:
        """Define spec for the environment."""

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

        self.observation_spec = Composite(
            {
                "step_count_seq": Bounded(
                    low=-1,  # -1 means unobserved
                    high=self.num_steps,
                    shape=torch.Size((*self.batch_size, self.num_steps + 1)),
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
                "game_id": Unbounded(
                    shape=torch.Size((*self.batch_size, 16)),  # UUID as 16 bytes
                    dtype=torch.uint8,
                    device=self.device,
                ),
                "actions_seq": Bounded(
                    low=-1,  # -1 means unobserved
                    high=self.action_size - 1,
                    shape=torch.Size((*self.batch_size, self.num_defenders + self.num_attackers, self.num_steps + 1)),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "actions_mask": Bounded(
                    low=0,
                    high=1,
                    shape=torch.Size((*self.batch_size, self.num_defenders + self.num_attackers, self.action_size)),
                    dtype=torch.bool,
                    device=self.device,
                ),
                # We assume that all instances of attackers and defenders share the same view
                "graph_x_seq": Unbounded(
                    shape=torch.Size((*self.batch_size, 2, self.num_steps + 1, self.map.num_nodes, self.graph_x_size)),
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
                **self._get_observation_spec(),
            },
            shape=self.batch_size,
            device=self.device,
        )

        self.action_spec = Bounded(
            low=0,
            high=self.action_size - 1,
            shape=torch.Size((*self.batch_size, self.num_defenders + self.num_attackers)),
            dtype=torch.int32,
            device=self.device,
        )

        # Reward for both defender and attacker
        self.reward_spec = Unbounded(
            shape=torch.Size((*self.batch_size, 2)),
            dtype=torch.float32,
            device=self.device,
        )

        self.done_spec = Composite(
            {
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

    @abstractmethod
    def _impl_reset(self) -> dict[str, torch.Tensor]:
        """
        Reset all custom environment variables and return the initial state.
        """

    @abstractmethod
    def _impl_step(
        self, tensordict: TensorDictBase, rewards: torch.Tensor, is_truncated: torch.Tensor, is_terminated: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Perform a step in the environment. rewards, is_truncated, and is_terminated are updated in-place.
        This method should be implemented by subclasses to define the environment's dynamics.
        """

    @abstractmethod
    def _get_graph_x(self, **kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the node features tensor `x` for the environment's graph.
        This method should be implemented by subclasses to provide the node features.
        """

    @final
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment and return the initial state.
        This method should be called to initialize the environment before starting an episode.
        """
        self.step_count = torch.zeros((*self.batch_size, 1), dtype=torch.int32, device=self.device)
        step_count_seq = torch.full((*self.batch_size, self.num_steps + 1), -1, dtype=torch.int32, device=self.device)
        step_count_seq[..., -1] = 0  # Set the last step count to 0

        # Generate game_id for each item in the batch
        if self.batch_size == torch.Size([]):  # Single environment case
            self.game_id = torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8, device=self.device)
        else:  # Batched environment case
            batch_uuids = [
                torch.tensor(list(uuid.uuid4().bytes), dtype=torch.uint8, device=self.device)
                for _ in range(self.batch_size[0])
            ]
            self.game_id = torch.stack(batch_uuids, dim=0)

        reset_output = self._impl_reset()
        graph_x_seq = torch.full(
            (*self.batch_size, 2, self.num_steps + 1, self.map.num_nodes, self.graph_x_size),
            -1.0,
            dtype=torch.float32,
            device=self.device,
        )
        graph_x_seq[..., -1, :, :] = self._get_graph_x(**reset_output)

        return TensorDict(
            {
                "step_count_seq": step_count_seq,
                "step_count": self.step_count.clone(),
                "actions_seq": torch.full((*self.batch_size, self.num_defenders + self.num_attackers, self.num_steps + 1), -1, dtype=torch.int32, device=self.device),
                "actions_mask": self.actions_mask,
                "game_id": self.game_id.clone(),
                "graph_x_seq": graph_x_seq,
                "graph_edge_index": self.map.edge_index.clone(),
                **reset_output,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    @final
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        assert tensordict.batch_size == torch.Size(()), "Batch size must be empty for step method."
        actions = tensordict["action"]
        previous_actions_seq = tensordict["actions_seq"][..., 1:]
        previous_step_count_seq = tensordict["step_count_seq"][..., 1:]
        previous_graph_x_seq = tensordict["graph_x_seq"][..., 1:, :, :]

        rewards = torch.zeros((*self.batch_size, 2), dtype=torch.float32, device=self.device)
        is_truncated = torch.zeros((1,), dtype=torch.bool, device=self.device)
        is_terminated = torch.zeros((1,), dtype=torch.bool, device=self.device)

        step_output = self._impl_step(tensordict, rewards, is_truncated, is_terminated)

        self.step_count += 1
        if self.step_count >= self.num_steps:
            is_truncated[0] = True

        graph_x = self._get_graph_x(**step_output)

        return TensorDict(
            {
                "step_count": self.step_count.clone(),
                "step_count_seq": torch.cat([previous_step_count_seq, self.step_count.clone()], dim=-1),
                "actions_mask": self.actions_mask,
                "actions_seq": torch.cat([previous_actions_seq, actions.unsqueeze(-1)], dim=-1),
                "reward": rewards,
                "truncated": is_truncated,
                "terminated": is_terminated,
                "graph_x_seq": torch.cat([
                    previous_graph_x_seq,
                    graph_x.unsqueeze(-3),
                ], dim=-3),
                "game_id": self.game_id.clone(),
                "graph_edge_index": self.map.edge_index.clone(),
                **step_output,
            },
            batch_size=self.batch_size,
            device=self.device,
        )
