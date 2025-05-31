from torch import nn

from .base import BaseAgent


class AgentGenerator:
    def __init__(
        self,
        agent_class: type[BaseAgent],
        kwargs: dict,
    ) -> None:
        self.agent_class = agent_class
        self.kwargs = kwargs

    def __call__(self, **kwargs) -> BaseAgent:
        return self.agent_class(**{**self.kwargs, **kwargs})
