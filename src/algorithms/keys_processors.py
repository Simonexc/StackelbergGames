from abc import ABC, abstractmethod
from typing import Type

import torch
from torchrl.envs import EnvBase

from environments.flipit_geometric import FlipItEnv


class TensorDictKeyExtractorBase(ABC):
    KEY: str

    def __init__(self, player_type: int, env: EnvBase) -> None:
        self._player_type = player_type
        self._env = env

        assert self.KEY in self._env.observation_spec, f"Key {self.KEY} not found in observation spec."

    @property
    def expected_size(self) -> int:
        """
        Expected size of the tensor extracted by this key extractor.
        """
        return self._env.observation_spec[self.KEY].shape[-1]

    @abstractmethod
    def process(self, value: torch.Tensor) -> torch.Tensor:
        """
        Process the value extracted from the tensordict and normalize it.
        Should be implemented by subclasses.
        """


class StepCountExtractor(TensorDictKeyExtractorBase):
    KEY = "step_count"

    def __init__(self, player_type: int, env: FlipItEnv) -> None:
        super().__init__(player_type, env)

        # assert isinstance(self._env, FlipItEnv), "StepCountExtractor is only compatible with FlipItEnv."

    @property
    def expected_size(self) -> int:
        return 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        assert value.shape[-1] == self.expected_size, f"Expected size {self.expected_size}, but got {value.shape[-1]}."
        return value / self._env.num_steps


class StepCountSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "step_count_seq"

    def __init__(self, player_type: int, env: FlipItEnv) -> None:
        super().__init__(player_type, env)

        #assert isinstance(self._env, FlipItEnv), "StepCountSeqExtractor is only compatible with FlipItEnv."

    @property
    def expected_size(self) -> int:
        return 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(-1) / self._env.num_steps - 0.5


class ObservedNodeOwnersSingleExtractor(TensorDictKeyExtractorBase):
    KEY = "observed_node_owners"

    def __init__(self, player_type: int, env: FlipItEnv) -> None:
        super().__init__(player_type, env)

        assert isinstance(self._env, FlipItEnv), "ObservedNodeOwnersSingleExtractor is only compatible with FlipItEnv."

    def process(self, value: torch.Tensor) -> torch.Tensor:
        assert value.shape[-1] == self.expected_size, f"Expected size {self.expected_size}, but got {value.shape[-1]}."

        return value[..., self._player_type, -1, :].float()


class ObservedNodeOwnersLastExtractor(TensorDictKeyExtractorBase):
    KEY = "observed_node_owners"

    def __init__(self, player_type: int, env: FlipItEnv) -> None:
        super().__init__(player_type, env)

        assert isinstance(self._env, FlipItEnv), "ObservedNodeOwnersExtractor is only compatible with FlipItEnv."

    def process(self, value: torch.Tensor) -> torch.Tensor:
        assert value.shape[-1] == self.expected_size, f"Expected size {self.expected_size}, but got {value.shape[-1]}."

        return value[..., self._player_type, -1, :].float()


class ObservedNodeOwnersExtractor(TensorDictKeyExtractorBase):
    KEY = "observed_node_owners"

    def __init__(self, player_type: int, env: FlipItEnv) -> None:
        super().__init__(player_type, env)

        assert isinstance(self._env, FlipItEnv), "ObservedNodeOwnersExtractor is only compatible with FlipItEnv."

    def process(self, value: torch.Tensor) -> torch.Tensor:
        assert value.shape[-1] == self.expected_size, f"Expected size {self.expected_size}, but got {value.shape[-1]}."

        return value[..., self._player_type, :, :].float()


class LastActionExtractor(TensorDictKeyExtractorBase):
    KEY = "actions_seq"

    @property
    def expected_size(self) -> int:
        return self._env.action_size + 1  # +1 for the "no action" case

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode actions
        actions_seq = (value[..., self._player_type, -1] + 1).long()  # Shift actions to start from 0
        actions_seq_one_hot = torch.nn.functional.one_hot(actions_seq, num_classes=self.expected_size).float()

        return actions_seq_one_hot


class ActionsExtractor(TensorDictKeyExtractorBase):
    KEY = "actions_seq"

    @property
    def expected_size(self) -> int:
        return self._env.action_size + 1  # +1 for the "no action" case

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode actions
        actions_seq = (value[..., self._player_type, :] + 1).long()  # Shift actions to start from 0
        actions_seq_one_hot = torch.nn.functional.one_hot(actions_seq, num_classes=self.expected_size).float()

        return actions_seq_one_hot


class PositionLastExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        return self._env.map.num_nodes + 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        position_seq = (value[..., self._player_type, -1] + 1).long()  # Shift positions to start from 0
        position_seq = torch.nn.functional.one_hot(position_seq, num_classes=self.expected_size).float()

        return position_seq


class PositionIntLastExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        return 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., self._player_type, -1].unsqueeze(-1)


class PositionIntSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        return 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., self._player_type, :].unsqueeze(-1)


class GraphEdgeIndexExtractor(TensorDictKeyExtractorBase):
    KEY = "graph_edge_index"

    @property
    def expected_size(self) -> int:
        return self._env.map.edge_index.shape[1]

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value


class TrackValueLastExtractor(TensorDictKeyExtractorBase):
    KEY = "track_value"

    @property
    def expected_size(self) -> int:
        return self._env.map.num_nodes

    def process(self, value: torch.Tensor) -> torch.Tensor:
        track_value = value[..., self._player_type, -1, :].float() / self._env.num_steps

        return track_value


class AvailableMovesLastExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        return (self._env.map.num_nodes + 1) * 4

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        available_moves = (value[..., self._player_type, -1, :] + 1).long()  # Shift positions to start from 0
        available_moves = torch.nn.functional.one_hot(available_moves, num_classes=self._env.map.num_nodes + 1).float()

        return available_moves.view(*available_moves.shape[:-2], self.expected_size)  # Flatten to (batch_size, num_nodes * 4)


class AvailableMovesIntExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        return 4

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., self._player_type, -1, :]


class AvailableMovesIntSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        return 4

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # Extract the last available moves
        available_moves = value[..., self._player_type, :, :]
        return available_moves


class NodeRewardInfoLastExtractor(TensorDictKeyExtractorBase):
    KEY = "node_reward_info"

    @property
    def expected_size(self) -> int:
        return 2

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # Extract the last node reward info
        node_reward_info = value[..., self._player_type, -1, :].float()

        return node_reward_info


class GraphXExtractor(TensorDictKeyExtractorBase):
    KEY = "graph_x"

    @property
    def expected_size(self) -> int:
        return self._env.graph_x_size

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., self._player_type, :, :]


class GraphXSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "graph_x_seq"

    @property
    def expected_size(self) -> int:
        return self._env.graph_x_size

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., self._player_type, :, :, :]


class CombinedExtractor:
    def __init__(self, player_type: int, env: FlipItEnv, actions_map: dict[str, list[Type[TensorDictKeyExtractorBase]]]) -> None:
        self._player_type = player_type
        self._env = env
        self._actions_map = {
            key: [action(player_type, env) for action in actions]
            for key, actions in actions_map.items()
        }

    @property
    def in_keys(self) -> list[str]:
        keys = []
        for actions in self._actions_map.values():
            for action in actions:
                if action.KEY not in keys:
                    keys.append(action.KEY)
        return keys

    @property
    def input_size(self) -> dict[str, int]:
        """
        Calculate the total expected size of the concatenated tensors.
        """
        return {
            key: sum(action.expected_size for action in actions)
            for key, actions in self._actions_map.items()
        }

    def process(self, *args: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process the values extracted from the tensordict and concatenate them.
        """
        processed_values: dict[str, torch.Tensor] = {}
        key_to_args = {key: arg for key, arg in zip(self.in_keys, args)}
        for key, actions in self._actions_map.items():
            processed_values[key] = torch.cat([action.process(key_to_args[action.KEY]) for action in actions], dim=-1)

        return processed_values
