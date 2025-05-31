from torchrl.modules import ActorValueOperator


class TrainAttacker:
    def __init__(
        self,
        defender_policy: ActorValueOperator,
        attacker_policy: ActorValueOperator,

    ) -> None:
        self._defender_policy = defender_policy
        self._attacker_policy = attacker_policy

