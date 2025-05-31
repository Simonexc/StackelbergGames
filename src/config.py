from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass

import torch


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
    num_steps: int
    path_to_map: str


@dataclass
class CoevoSGConfig(FromDictMixin):
    generations: int = 1000  # total max generations
    gen_per_switch: int = 20
    elite_size: int = 10
    crossover_prob: float = 0.8
    mutation_prob: float = 0.5
    selection_pressure: float = 0.9
    attacker_eval_top_n: int = 10
    no_improvement_limit: int = 100
