import networkx as nx
import numpy as np
import random
import torch
from config import Player

from environments.flipit_utils import Action, ActionTargetPair, PlayerTargetPair


class FlipItMap:
    """
    Generates a FlipIt graph.
    """

    def __init__(self, num_nodes: int, seed: int) -> None:
        self.num_nodes = num_nodes
        self.seed = seed
        self.adjacency_matrix, self.entry_nodes = self._graph_generator()

        generator = torch.Generator().manual_seed(self.seed)
        self.node_rewards = torch.rand(self.num_nodes, generator=generator, dtype=torch.float32)
        self.node_costs = -torch.rand(self.num_nodes, generator=generator, dtype=torch.float32)

    @classmethod
    def load(cls, path: str) -> "FlipItMap":
        """
        Loads a FlipItMap from a file.
        """
        return torch.load(path, weights_only=False)

    def save(self, path: str) -> None:
        """
        Saves the FlipItMap to a file.
        """
        torch.save(self, path)

    def _graph_generator(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates a default Watts-Strogatz graph."""
        random.seed(self.seed)
        k = 3  # Average degree from paper
        p = 0.1  # Rewiring probability

        # 1. Generate the base undirected Watts-Strogatz graph
        # Note: nx.watts_strogatz_graph uses its own seed for internal generation,
        # but we'll use our 'rng' for subsequent choices (like edge direction).
        G_undirected = nx.watts_strogatz_graph(self.num_nodes, k, p, seed=self.seed)

        # 2. Create a new empty directed graph
        G = nx.DiGraph()
        # Add all nodes from the undirected graph to the directed graph
        # This ensures nodes without edges are included.
        G.add_nodes_from(G_undirected.nodes())

        # 3. Iterate through undirected edges and add one directed edge randomly
        for u, v in G_undirected.edges():
            # Use the seeded RNG to choose the direction for this edge
            if random.choice([True, False]):
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

        # 4. Ensure graph is weakly connected if possible, otherwise add edges
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            if len(components) > 1:
                # Optional: Sort components for slightly more deterministic connection process
                # components.sort(key=min) # Example: sort by minimum node ID in component
                for i in range(len(components) - 1):
                    # Use the seeded RNG to choose nodes for connecting components
                    u = random.choice(list(components[i]))
                    v = random.choice(list(components[i + 1]))
                    # Only add if no edge exists in either direction (unlikely but safe)
                    if not G.has_edge(u, v) and not G.has_edge(v, u):
                        G.add_edge(u, v)  # Add a single directed edge

        # 5. Convert to adjacency matrix
        adjacency_matrix = torch.from_numpy(nx.to_numpy_array(G, nodelist=G.nodes(), dtype=np.bool_))

        # Select entry nodes (e.g., 10% of nodes, minimum 2)
        num_entry = max(2, self.num_nodes // 10)
        entry_nodes = torch.zeros(self.num_nodes, dtype=torch.bool)
        entry_nodes[random.sample(range(self.num_nodes), num_entry)] = True

        return adjacency_matrix, entry_nodes


class FlipItEnv:
    """
    FlipIt Game Environment.

    Players: Defender (0) and Attacker (1)
    Actions:
        - Flip a node: Attempt to take control. Requires reachability. Costs node-specific value.
        - Observe a node: Reveal the current owner. Costs nothing. Does not require reachability.
    State: Node ownership, current time step.
    Rewards: Sum of rewards for controlled nodes minus costs of attempted flips over time steps.
    Termination: Game ends after a fixed number of time steps 'm'.
    Observability: The environment returns the full state, but the 'info' dict
                   contains results of 'observe' actions, facilitating partial observability
                   agent implementations.
    """

    def __init__(
        self,
        num_nodes: int,
        num_steps: int,
        adjacency_matrix: torch.Tensor,
        entry_nodes: torch.Tensor,
        node_rewards: torch.Tensor,
        node_costs: torch.Tensor,
    ) -> None:
        self.n = num_nodes
        if num_nodes < 3:
            raise ValueError("Number of nodes must be at least 3.")
        self.m = num_steps

        # --- Game State ---
        self.adjacency_matrix = adjacency_matrix  # Adjacency matrix of the graph
        self.entry_nodes = entry_nodes  # Bool array indicating iuf a node is entry node
        self.node_rewards = node_rewards  # Reward for controlling node i
        self.node_costs = node_costs  # Cost for attempting flip on node i (< 0)
        self.node_owners = torch.zeros(self.n, dtype=torch.bool)  # Current owner of node i (DEFENDER or ATTACKER)
        self.current_step: int = 0
        self.cumulative_rewards = torch.zeros(2, dtype=torch.float32)  # Cumulative rewards for each player
        self.action_space = torch.cartesian_prod(torch.tensor([0, 1]), torch.arange(self.n)).to(dtype=torch.int32)  # All possible action-target pairs

        self.history = (
            torch.zeros((self.m, 2), dtype=torch.bool),
            torch.zeros((self.m, 2), dtype=torch.int32),
            torch.zeros((self.m, 2), dtype=torch.float32),
        )

        self.reset()

    @classmethod
    def from_map(cls, num_steps: int, flipit_map: FlipItMap) -> "FlipItEnv":
        """
        Creates a FlipIt environment using the provided FlipItMap.
        """
        return cls(
            num_nodes=flipit_map.num_nodes,
            num_steps=num_steps,
            adjacency_matrix=flipit_map.adjacency_matrix,
            entry_nodes=flipit_map.entry_nodes,
            node_rewards=flipit_map.node_rewards,
            node_costs=flipit_map.node_costs,
        )

    def reset(self) -> None:
        """
        Resets the environment to the initial state.
        """

        self.current_step = 0
        self.node_owners = torch.zeros(self.n, dtype=torch.bool)  # Defender starts owning all
        self.cumulative_rewards = torch.zeros(2, dtype=torch.float32)  # Reset cumulative rewards

        self.history = (
            torch.zeros((self.m, 2), dtype=torch.bool),
            torch.zeros((self.m, 2), dtype=torch.int32),
            torch.zeros((self.m, 2), dtype=torch.float32),
        )

    @property
    def game_ended(self) -> bool:
        """Checks if the game has ended."""
        return self.current_step >= self.m

    def _node_exists(self, node: int) -> bool:
        """Checks if a node exists in the graph."""
        return 0 <= node < self.n

    def predecessors(self, node: int) -> torch.Tensor:
        return self.adjacency_matrix[:, node]

    def successors(self, node: int) -> torch.Tensor:
        return self.adjacency_matrix[node, :]

    def _is_reachable_for_attacker(self, target_node: int) -> bool:
        """
        Checks if the attacker can attempt a flip on the target node.
        """
        if self.entry_nodes[target_node]:  # Rule 1.e.i
            return True

        return (self.node_owners & self.predecessors(target_node)).any().item()  # Rule 1.e.ii

    def _is_reachable_for_defender(self, target_node: int) -> bool:
        """
        Checks if the defender can attempt a flip on the target node.
        """
        return (~self.node_owners & self.successors(target_node)).any().item()  # Rule 1.e.iii

    def _is_reachable(self, player: Player, target_node: int) -> bool:
        """
        Checks if a player can attempt a flip on the target node based on FlipIt rules.
        """
        if not self._node_exists(target_node):
            raise ValueError(f"Node {target_node} does not exist in the graph.")

        if player == Player.defender:
            return self._is_reachable_for_defender(target_node)
        elif player == Player.attacker:
            return self._is_reachable_for_attacker(target_node)
        else:
            raise ValueError(f"Invalid player {player}.")

    @property
    def game_info(self) -> dict[str, int | torch.Tensor]:
        return {
            "num_nodes": self.n,
            "num_steps": self.m,
            "adjacency_matrix": self.adjacency_matrix,
            "entry_nodes": self.entry_nodes,
            "node_rewards": self.node_rewards,
            "node_costs": self.node_costs,
            "action_space": self.action_space,
        }

    def step(
        self, defender: ActionTargetPair, attacker: ActionTargetPair
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, int, bool]:
        """
        Executes one time step in the game.

        Args:
            defender (ActionTargetPair): Action and target node chosen by the defender.
            attacker (ActionTargetPair): Action and target node chosen by the attacker.

        Returns:
            Tuple containing:
            - observations (Dict[Player, PlayerTargetPair]): Observations for each player after the step.
            - rewards (Dict[Player, float]): Rewards obtained by each player in this step.
            - current_step (int): The current time step in the game.
            - game_ended (bool): Indicates if the game has ended.
        """
        if self.game_ended:
            raise RuntimeError("Game has ended. Please reset the environment.")

        # defender, attacker
        step_costs = torch.zeros(2, dtype=torch.float32)  # Costs for defender and attacker
        observe_results = (
            torch.zeros(2, dtype=torch.bool),
            torch.full((2,), -1, dtype=torch.int32),
        )

        # --- Process Actions and Calculate Costs/Observe ---
        valid_flips: list[PlayerTargetPair] = []
        valid_observations: list[PlayerTargetPair] = []
        target_nodes = torch.tensor([defender.target_node, attacker.target_node], dtype=torch.int32)
        self.history[0][self.current_step] = self.node_owners[target_nodes]
        self.history[1][self.current_step] = target_nodes
        self.history[2][self.current_step] = self.cumulative_rewards.clone()

        # Validate actions
        for player, (action, target_node) in zip([Player.defender, Player.attacker], [defender, attacker]):
            if target_node < 0 or target_node >= self.n:
                raise RuntimeError(f"Player {player} chose invalid node {target_node}. Action ignored.")

            if action == Action.flip:
                step_costs[player.value] += self.node_costs[target_node]  # Cost for flip attempt (Rule 1.a)
                if self._is_reachable(player, target_node):  # Rule 1.e
                    # Check if node is being flipped by both players
                    if len(valid_flips) == 1 and valid_flips[0].target_node == target_node:  # Rule 1.c
                        valid_flips.clear()  # Ignore both
                    else:
                        valid_flips.append(PlayerTargetPair(player, target_node))
            elif action == Action.observe:
                # Observe action costs nothing
                valid_observations.append(PlayerTargetPair(player, target_node))
            else:
                raise RuntimeError(f"Player {player} chose invalid action type {action}. Action ignored.")

        # Perform flips
        for player, target_node in valid_flips:
            self.node_owners[target_node] = bool(player.value)

        # Perform observations
        for player, target_node in valid_observations:
            observe_results[0][player.value] = self.node_owners[target_node]
            observe_results[1][player.value] = target_node

        # Calculate step rewards based on new ownership
        step_rewards = torch.tensor([
            self.node_rewards[~self.node_owners].sum(),
            self.node_rewards[self.node_owners].sum(),
        ], dtype=torch.float32)

        # Update cumulative rewards and time step
        total_step_rewards = step_rewards + step_costs
        self.cumulative_rewards += total_step_rewards

        self.current_step += 1

        return observe_results, total_step_rewards, self.current_step, self.game_ended

    def undo_step(self) -> None:
        self.current_step -= 1
        self.cumulative_rewards = self.history[2][self.current_step].clone()
        self.node_owners[self.history[1][self.current_step]] = self.history[0][self.current_step].clone()
