from enum import Enum
from typing import NamedTuple

from .flipit_geometric import FlipItEnv, FlipItMap
from .poachers import PoachersMap, PoachersEnv


class EnvPair(NamedTuple):
    env_class: type[FlipItEnv | PoachersEnv]
    map_class: type[FlipItMap | PoachersMap]


class EnvMapper(Enum):
    flipit = EnvPair(
        env_class=FlipItEnv,
        map_class=FlipItMap
    )
    poachers = EnvPair(
        env_class=PoachersEnv,
        map_class=PoachersMap,
    )

    @classmethod
    def from_name(cls, env_name: str) -> "EnvMapper":
        for env in cls:
            if env.name == env_name:
                return env

        raise ValueError(f"Environment '{env_name}' not found.")
