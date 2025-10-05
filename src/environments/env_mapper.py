from enum import Enum
from typing import NamedTuple

from environments.flipit_geometric import FlipItEnv, FlipItMap
from environments.poachers import PoachersEnv, PoachersMap
from environments.police import PoliceEnv, PoliceMap


class EnvPair(NamedTuple):
    env_class: type[FlipItEnv | PoachersEnv | PoliceEnv]
    map_class: type[FlipItMap | PoachersMap | PoliceMap]


class EnvMapper(Enum):
    flipit = EnvPair(
        env_class=FlipItEnv,
        map_class=FlipItMap
    )
    poachers = EnvPair(
        env_class=PoachersEnv,
        map_class=PoachersMap,
    )
    police = EnvPair(
        env_class=PoliceEnv,
        map_class=PoliceMap,
    )

    @classmethod
    def from_name(cls, env_name: str) -> "EnvMapper":
        for env in cls:
            if env.name == env_name:
                return env

        raise ValueError(f"Environment '{env_name}' not found.")
