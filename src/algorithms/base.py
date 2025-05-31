from abc import ABC, abstractmethod
import os

import torch
from torch import nn
from tensordict import TensorDictBase
from torchrl.data import ReplayBuffer

from config import Player


class BaseAgent(nn.Module, ABC):
    def __init__(
        self,
        action_size: int,
        player_type: int,
        embedding_size: int,
        device: torch.device | str,
        run_name: str,
        agent_id: int | None = None,
    ) -> None:
        super().__init__()

        self.action_size = action_size
        self.player_type = player_type
        self.embedding_size = embedding_size
        self._device = device
        self.player_name = Player(player_type).name
        self.run_name = run_name
        self.agent_id = agent_id

        self.num_steps: int = 0
        self.num_epochs: int = 0
        self.num_cycle: int = 0
        self.currently_training = False

    def save(self) -> None:
        run_folder = os.path.join("saved_models", self.run_name, self.player_name)
        save_path = os.path.join(run_folder, f"agent_{self.agent_id or 0}.pth")
        os.makedirs(run_folder, exist_ok=True)
        torch.save(self.state_dict(), save_path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self._device))

    @abstractmethod
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Forward pass of the agent.
        Should be implemented by subclasses.
        """


class BaseTrainableAgent(BaseAgent, ABC):
    @abstractmethod
    def train_cycle(self, tensordict_data: TensorDictBase, replay_buffer: ReplayBuffer, cycle_num: int) -> float:
        """
        Train the agent for one cycle.
        Should be implemented by subclasses.
        """
