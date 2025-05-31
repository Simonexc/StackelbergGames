import time

from environments.flipit import FlipItEnv, default_graph_generator
from environments.flipit_utils import Action, ActionTargetPair, PlayerTargetPair
from strategies import Policy, OracleAttackerPolicy
import torch

NUM_NODES = 10
NUM_STEPS = 5
SEED = 42

adjacency_matrix, entry_nodes = default_graph_generator(num_nodes=NUM_NODES, seed=SEED)
node_rewards = torch.rand(NUM_NODES)
node_costs = -torch.rand(NUM_NODES)

env = FlipItEnv(
    num_nodes=NUM_NODES,
    num_steps=NUM_STEPS,
    adjacency_matrix=adjacency_matrix,
    entry_nodes=entry_nodes,
    node_rewards=node_rewards,
    node_costs=node_costs,
)


class MyPolicy(Policy):
    def get_probs(self, extended_observations: list[PlayerTargetPair | None] | None = None) -> list[float]:
        return [0.5 if at.target_node in [0, 1] and at.action == Action.flip else 0.0 for at in self.action_space]


action_space = [ActionTargetPair(Action.flip, i) for i in range(NUM_NODES)] + [ActionTargetPair(Action.observe, i) for i
                                                                               in range(NUM_NODES)]
defender_policy = MyPolicy(action_space)

attacker_policy = OracleAttackerPolicy(action_space, defender_policy, env)

start_time = time.time()
for i in range(1):
    attacker_policy.sample_action()
print("Time taken to sample action:", (time.time() - start_time)/1)
