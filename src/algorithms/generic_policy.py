import torch
from torch import nn
from tensordict import TensorDictBase

from .base import BaseAgent


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
