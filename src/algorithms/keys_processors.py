from abc import ABC, abstractmethod
from typing import Type

import torch
from environments.base_env import EnvironmentBase

from environments.flipit_geometric import FlipItEnv


class TensorDictKeyExtractorBase(ABC):
    KEY: str

    def __init__(self, player_type: int, env: EnvironmentBase) -> None:
        self._player_type = player_type
        self._env = env

        # assert self.KEY in self._env.observation_spec, f"Key {self.KEY} not found in observation spec."

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


class AvailableMovesIntForFlipItExtractor(TensorDictKeyExtractorBase):
    KEY = "observed_node_owners"

    @property
    def expected_size(self) -> int:
        return self._env.map.num_nodes - 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim <= 3:
            return torch.arange(1, value.shape[-1]).to(torch.long).to(value.device)
        return torch.arange(1, value.shape[-1]).to(torch.long).repeat(*value.shape[:-3], 1).to(value.device)


class AvailableMovesIntForFlipItSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "observed_node_owners"

    @property
    def expected_size(self) -> int:
        return self._env.map.num_nodes - 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim <= 3:
            return torch.arange(1, value.shape[-1]).to(torch.long).repeat(value.shape[-2], 1).to(value.device)
        return torch.arange(1, value.shape[-1]).to(torch.long).repeat(*value.shape[:-3], value.shape[-2], 1).to(value.device)


class PositionIntForFlipItExtractor(TensorDictKeyExtractorBase):
    KEY = "observed_node_owners"

    @property
    def expected_size(self) -> int:
        return 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim <= 3:
            return torch.tensor([0], dtype=torch.long).to(value.device)
        return torch.tensor([0], dtype=torch.long).repeat(*value.shape[:-3], 1).to(value.device)


class PositionIntForFlipItSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "observed_node_owners"

    @property
    def expected_size(self) -> int:
        return 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim <= 3:
            return torch.tensor([0], dtype=torch.long).repeat(value.shape[-2], 1).to(value.device)
        return torch.tensor([0], dtype=torch.long).repeat(*value.shape[:-3], value.shape[-2], 1).to(value.device)


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
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return (self._env.action_size + 1) * num  # +1 for the "no action" case

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode actions
        if self._player_type == 0:
            actions_seq = (value[..., :self._env.num_defenders, -1] + 1).long()  # Shift actions to start from 0
        else:
            actions_seq = (value[..., self._env.num_defenders:, -1] + 1).long()  # Shift actions to start from 0
        actions_seq_one_hot = torch.nn.functional.one_hot(actions_seq, num_classes=self._env.action_size + 1).reshape((*actions_seq.shape[:-1], -1)).float()
        return actions_seq_one_hot


class ActionsExtractor(TensorDictKeyExtractorBase):
    KEY = "actions_seq"

    @property
    def expected_size(self) -> int:
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return (self._env.action_size + 1) * num  # +1 for the "no action" case

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode actions
        if self._player_type == 0:
            actions_seq = (value[..., :self._env.num_defenders, :] + 1).long()  # Shift actions to start from 0
        else:
            actions_seq = (value[..., self._env.num_defenders:, :] + 1).long()  # Shift actions to start from 0
        actions_seq_one_hot = torch.nn.functional.one_hot(actions_seq.transpose(-1, -2), num_classes=self._env.action_size + 1)

        return actions_seq_one_hot.reshape((*actions_seq.shape[:-2], actions_seq.shape[-1], -1)).float()


class PositionLastExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return (self._env.map.num_nodes + 1) * num

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        if self._player_type == 0:
            position_seq = (value[..., :self._env.num_defenders, -1] + 1).long()  # Shift actions to start from 0
        else:
            position_seq = (value[..., self._env.num_defenders:, -1] + 1).long()  # Shift actions to start from 0
        position_seq_one_hot = torch.nn.functional.one_hot(position_seq, num_classes=self._env.map.num_nodes + 1).reshape((*position_seq.shape[:-1], -1)).float()
        return position_seq_one_hot


class PositionSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return (self._env.map.num_nodes + 1) * num

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        if self._player_type == 0:
            position_seq = (value[..., :self._env.num_defenders, :] + 1).long()  # Shift actions to start from 0
        else:
            position_seq = (value[..., self._env.num_defenders:, :] + 1).long()  # Shift actions to start from 0
        position_seq_one_hot = torch.nn.functional.one_hot(position_seq.transpose(-1, -2), num_classes=self._env.map.num_nodes + 1).reshape((*position_seq.shape[:-2], position_seq.shape[-1], -1)).float()
        return position_seq_one_hot


class AllPositionLastExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        return (self._env.map.num_nodes + 1) * (self._env.num_defenders + self._env.num_attackers)

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        position_seq = (value[..., :, -1] + 1).long()  # Shift actions to start from 0
        position_seq_one_hot = torch.nn.functional.one_hot(position_seq, num_classes=self._env.map.num_nodes + 1).reshape((*position_seq.shape[:-1], -1)).float()
        return position_seq_one_hot


class AllPositionSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        return (self._env.map.num_nodes + 1) * (self._env.num_defenders + self._env.num_attackers)

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        position_seq = (value + 1).long()  # Shift actions to start from 0
        position_seq_one_hot = torch.nn.functional.one_hot(position_seq.transpose(-1, -2), num_classes=self._env.map.num_nodes + 1).reshape((*position_seq.shape[:-2], position_seq.shape[-1], -1)).float()
        return position_seq_one_hot


class PositionIntLastExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return num

    def process(self, value: torch.Tensor) -> torch.Tensor:
        if self._player_type == 0:
            return value[..., :self._env.num_defenders, -1]
        return value[..., self._env.num_defenders:, -1]


class AllPositionIntLastExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        return self._env.num_defenders + self._env.num_attackers

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., -1]


class PositionIntSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return num

    def process(self, value: torch.Tensor) -> torch.Tensor:
        if self._player_type == 0:
            return value[..., :self._env.num_defenders, :].transpose(-1, -2)
        return value[..., self._env.num_defenders:, :].transpose(-1, -2)


class AllPositionIntSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "position_seq"

    @property
    def expected_size(self) -> int:
        return self._env.num_defenders + self._env.num_attackers

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value.transpose(-1, -2)


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
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return (self._env.map.num_nodes + 1) * 4 * num

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        if self._player_type == 0:
            available_moves = (value[..., :self._env.num_defenders, -1, :] + 1).long()  # Shift positions to start from 0
        else:
            available_moves = (value[..., self._env.num_defenders:, -1, :] + 1).long()  # Shift positions to start from 0

        # Flatten to (batch_size, (num_nodes+1) * 4 * num)
        available_moves_one_hot = torch.nn.functional.one_hot(available_moves, num_classes=self._env.map.num_nodes + 1).reshape(*available_moves.shape[:-2], -1).float()
        return available_moves_one_hot


class AvailableMovesSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return (self._env.map.num_nodes + 1) * 4 * num

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        if self._player_type == 0:
            available_moves = (value[..., :self._env.num_defenders, :, :] + 1).long()  # Shift positions to start from 0
        else:
            available_moves = (value[..., self._env.num_defenders:, :, :] + 1).long()  # Shift positions to start from 0

        # Flatten to (batch_size, (num_nodes+1) * 4 * num)
        available_moves_one_hot = torch.nn.functional.one_hot(available_moves.transpose(-2, -3), num_classes=self._env.map.num_nodes + 1).reshape(*available_moves.shape[:-3], available_moves.shape[-2], -1).float()
        return available_moves_one_hot


class AllAvailableMovesLastExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        return (self._env.map.num_nodes + 1) * 4 * (self._env.num_defenders + self._env.num_attackers)

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        available_moves = (value[..., -1, :] + 1).long()  # Shift positions to start from 0

        # Flatten to (batch_size, (num_nodes+1) * 4 * num)
        available_moves_one_hot = torch.nn.functional.one_hot(available_moves, num_classes=self._env.map.num_nodes + 1).reshape(*available_moves.shape[:-2], -1).float()
        return available_moves_one_hot


class AllAvailableMovesSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        return (self._env.map.num_nodes + 1) * 4 * (self._env.num_defenders + self._env.num_attackers)

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # One-Hot encode position
        available_moves = (value + 1).long()  # Shift positions to start from 0

        # Flatten to (batch_size, (num_nodes+1) * 4 * num)
        available_moves_one_hot = torch.nn.functional.one_hot(available_moves.transpose(-2, -3), num_classes=self._env.map.num_nodes + 1).reshape(*available_moves.shape[:-3], available_moves.shape[-2], -1).float()
        return available_moves_one_hot


class AvailableMovesIntExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return 4 * num

    def process(self, value: torch.Tensor) -> torch.Tensor:
        if self._player_type == 0:
            available_moves = value[..., :self._env.num_defenders, -1, :]
        else:
            available_moves = value[..., self._env.num_defenders:, -1, :]

        return available_moves.reshape(*value.shape[:-3], -1)


class AllAvailableMovesIntExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        return 4 * (self._env.num_defenders + self._env.num_attackers)

    def process(self, value: torch.Tensor) -> torch.Tensor:
        available_moves = value[..., -1, :]

        return available_moves.reshape(*value.shape[:-3], -1)


class AvailableMovesIntSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        num = self._env.num_defenders if self._player_type == 0 else self._env.num_attackers
        return 4 * num

    def process(self, value: torch.Tensor) -> torch.Tensor:
        if self._player_type == 0:
            available_moves = value[..., :self._env.num_defenders, :, :]
        else:
            available_moves = value[..., self._env.num_defenders:, :, :]

        return available_moves.transpose(-2, -3).reshape(*value.shape[:-3], value.shape[-2], -1)


class AllAvailableMovesIntSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "available_moves"

    @property
    def expected_size(self) -> int:
        return 4 * (self._env.num_defenders + self._env.num_attackers)

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value.transpose(-2, -3).reshape(*value.shape[:-3], value.shape[-2], -1)


class NodeRewardInfoLastExtractor(TensorDictKeyExtractorBase):
    KEY = "node_reward_info"

    @property
    def expected_size(self) -> int:
        return 2

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # Extract the last node reward info
        node_reward_info = value[..., self._player_type, -1, :].float()

        return node_reward_info


class TargetsAttackedLastExtractor(TensorDictKeyExtractorBase):
    KEY = "targets_attacked_obs"

    @property
    def expected_size(self) -> int:
        return self._env.map.num_nodes

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # Extract the last targets attacked
        targets_attacked = value[..., self._player_type, -1, :].float()

        return targets_attacked


class CheckResultsLastExtractor(TensorDictKeyExtractorBase):
    KEY = "check_results"

    @property
    def expected_size(self) -> int:
        return self._env.map.num_nodes

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # Extract the last check results
        check_results = value[..., -1, :].float()

        return check_results


class CanAttackLastExtractor(TensorDictKeyExtractorBase):
    KEY = "can_attack_obs"

    @property
    def expected_size(self) -> int:
        return 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # Extract the last can_attack
        can_attack = value[..., -1].unsqueeze(-1).float()

        return can_attack


class CanAttackSeqExtractor(TensorDictKeyExtractorBase):
    KEY = "can_attack_obs"

    @property
    def expected_size(self) -> int:
        return 1

    def process(self, value: torch.Tensor) -> torch.Tensor:
        # Extract sequence of can_attack
        can_attack = value.float()

        return can_attack.unsqueeze(-1)


class GraphXExtractor(TensorDictKeyExtractorBase):
    KEY = "graph_x_seq"

    @property
    def expected_size(self) -> int:
        return self._env.graph_x_size

    def process(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., self._player_type, -1, :, :]


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
