import inspect
import sys
from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass

import torch

from algorithms import keys_processors
from environments.env_mapper import EnvMapper


class Player(Enum):
    defender = 0
    attacker = 1

    @classmethod
    def _missing_(cls, value: Any) -> Optional["Player"]:
        if isinstance(value, torch.Tensor):
            if value.dtype not in [torch.int32, torch.bool] or value.shape != torch.Size([]):
                raise ValueError(f"Player value must be a scalar tensor of type int32 or bool")

            value_int = int(value.item())

            for member in cls:
                if member.value == value_int:
                    return member

            raise ValueError(f"Invalid Player value: {value_int}. Must be 0 (defender) or 1 (attacker).")

        return None

    @classmethod
    def players(cls) -> list["Player"]:
        return [cls.defender, cls.attacker]

    @property
    def name(self) -> str:
        return "defender" if self == Player.defender else "attacker"

    @property
    def opponent(self) -> "Player":
        match self:
            case Player.defender:
                return Player.attacker
            case Player.attacker:
                return Player.defender
            case _:
                raise ValueError(f"Invalid player: {self}.")


class FromDictMixin:
    @classmethod
    def from_dict(cls, data: dict[str, Any], suffix: str = "") -> "FromDictMixin":
        keys = [key + suffix for key in cls.__dataclass_fields__.keys()]

        return cls(**{key.replace(suffix, ""): item for key, item in data.items() if key in keys})


@dataclass
class TrainingConfig(FromDictMixin):
    player_turns: int  # number of turns per player, starting from the defender
    total_steps_per_turn: int
    steps_per_batch: int  # we process one batch in each cycle
    epochs_per_batch: int
    sub_batch_size: int


@dataclass
class AgentNNConfig(FromDictMixin):
    embedding_size: int = 32


@dataclass
class LossConfig(FromDictMixin):
    clip_epsilon: float
    gamma: float
    lmbda: float
    entropy_eps: float
    critic_coef: float
    learning_rate: float
    max_grad_norm: float


@dataclass
class EnvConfig(FromDictMixin):
    env_name: str
    num_steps: int
    seed: int
    num_nodes: int

    def create(self, device: torch.device | str | None = None):
        env_pair = EnvMapper.from_name(self.env_name)
        map_obj = env_pair.value.map_class(self, device=device)
        env_obj = env_pair.value.env_class(self, map_obj, device=device)
        return map_obj, env_obj


@dataclass
class CoevoSGConfig(FromDictMixin):
    generations: int = 1000  # total max generations
    gen_per_switch: int = 20
    elite_size: int = 10
    crossover_prob: float = 0.8
    mutation_prob: float = 0.5
    selection_pressure: float = 0.9
    attacker_eval_top_n: int = 10
    no_improvement_limit: int = 40


@dataclass
class BackboneConfig(FromDictMixin):
    cls_name: str
    keys: dict[str, list[str]]
    d_model: int = 32 * 4
    num_head: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    hidden_size: int = 32
    embedding_cls_name: str | None = None

    def __post_init__(self) -> None:
        available_extractors: dict[str, type[keys_processors.TensorDictKeyExtractorBase]] = {
            name: cls
            for name, cls in inspect.getmembers(sys.modules[keys_processors.__name__], inspect.isclass)
            if issubclass(cls, keys_processors.TensorDictKeyExtractorBase) and cls is not keys_processors.TensorDictKeyExtractorBase
        }
        self.extractors: dict[str, list[type[keys_processors.TensorDictKeyExtractorBase]]] = {}

        for key, extractor_names in self.keys.items():
            curr_extractors: list[type[keys_processors.TensorDictKeyExtractorBase]] = []
            for extractor_name in extractor_names:
                if extractor_name not in available_extractors:
                    raise ValueError(f"Key '{extractor_name}' is not a valid extractor. Available keys: {list(available_extractors.keys())}")
                curr_extractors.append(available_extractors[extractor_name])

            self.extractors[key] = curr_extractors


@dataclass
class HeadConfig(FromDictMixin):
    hidden_size: int = 64
    num_heads: int = 1  # Number of actor heads (for multi-agent support)
