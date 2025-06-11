from abc import ABC, abstractmethod
import copy
import os
import multiprocessing
from functools import partial

import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
from tensordict import TensorDictBase
from tensordict.nn.probabilistic import (
    InteractionType as ExplorationType,
    interaction_type as get_interaction_type,
)

from .base import BaseAgent
from .generic_policy import CombinedPolicy
from config import CoevoSGConfig, Player

from environments.flipit_utils import generate_random_pure_strategy
from environments.flipit_geometric import FlipItEnv


class StrategyBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Forward pass of the strategy.
        Should return a TensorDict with action, logits, sample_log_prob, and embedding.
        """

    @abstractmethod
    def sample_pure_strategy(self) -> "PureStrategy":
        """
        Should return an instance of PureStrategy from the current strategy.
        """

    @abstractmethod
    def mutate(self) -> None:
        """
        Mutate the current strategy.
        Should be implemented by subclasses.
        """

    @abstractmethod
    def crossover(self, other: "StrategyBase") -> tuple["StrategyBase", "StrategyBase"]:
        """
        Crossover between two strategies.
        """

    @abstractmethod
    def copy(self) -> "StrategyBase":
        """
        Should return a copy of the current strategy.
        """

    @abstractmethod
    def simplify(self, min_prob_threshold: float = 1e-4, max_pure_strategies: int = 50) -> None:
        """
        Simplifies the mixed strategy by merging duplicates and removing low probability entries.
        """

    @abstractmethod
    def evaluate(self, env: FlipItEnv, opponent_population: list["StrategyBase"], top_n: int | None = None) -> None:
        """
        Evaluates the strategy against the environment and updates its fitness.
        Should be implemented by subclasses.
        """

    @abstractmethod
    def save(self, run_name: str, player_name: str) -> None:
        """
        Save the strategy to disk.
        Should be implemented by subclasses.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the strategy from disk.
        Should be implemented by subclasses.
        """

    @staticmethod
    def get_path_name(run_name: str, player_name: str) -> str:
        """
        Returns the path name for saving/loading the strategy.
        """
        run_folder = os.path.join("saved_models", run_name, player_name)
        save_path = os.path.join(run_folder, "agent_0.pth")
        os.makedirs(run_folder, exist_ok=True)
        return save_path


class PureStrategy(StrategyBase):
    def __init__(self, num_nodes: int, pure_strategy: torch.Tensor) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.pure_strategy = pure_strategy
        self.fitness = -float("inf")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = self.pure_strategy[tensordict["step_count"].item()]
        logits = torch.zeros(self.num_nodes*2, dtype=torch.float32, device=tensordict.device)
        logits[action] = 1.0
        sample_log_prob = torch.zeros(torch.Size(()), dtype=torch.float32, device=tensordict.device)
        embedding = torch.zeros(32, dtype=torch.float32, device=tensordict.device)

        tensordict.update({
            "action": action,
            "logits": logits,
            "sample_log_prob": sample_log_prob,
            "embedding": embedding,
        })
        return tensordict

    def sample_pure_strategy(self) -> "PureStrategy":
        return self

    def mutate(self) -> None:
        mutation_point = torch.randint(0, len(self.pure_strategy), torch.Size(())).item()
        self.pure_strategy[mutation_point:] = torch.randint(0, self.num_nodes, torch.Size((len(self.pure_strategy) - mutation_point,)), dtype=torch.int32)

    def crossover(self, other: "StrategyBase") -> tuple["StrategyBase", "StrategyBase"]:
        if not isinstance(other, PureStrategy):
            raise ValueError("Crossover can only be performed with another PureStrategy.")

        strat1 = self.pure_strategy
        strat2 = other.pure_strategy

        # One-point crossover for attacker pure strategies.
        if strat1.shape != strat2.shape:
            raise ValueError("Strategy lengths differ")

        if len(strat1) < 2:
            return PureStrategy(self.num_nodes, strat1.clone()), PureStrategy(self.num_nodes, strat2.clone())

        # Simple one-point crossover
        crossover_point = torch.randint(0, len(self.pure_strategy), torch.Size(())).item()
        child1_strategy = torch.cat((self.pure_strategy[:crossover_point], other.pure_strategy[crossover_point:]))
        child2_strategy = torch.cat((other.pure_strategy[:crossover_point], self.pure_strategy[crossover_point:]))

        return PureStrategy(self.num_nodes, child1_strategy), PureStrategy(self.num_nodes, child2_strategy)

    def copy(self) -> "PureStrategy":
        return PureStrategy(self.num_nodes, self.pure_strategy.clone())

    def simplify(self, min_prob_threshold: float = 1e-4, max_pure_strategies: int = 50) -> None:
        """ Pure strategies do not need simplification. """
        pass

    def evaluate(self, env: FlipItEnv, opponent_population: list["StrategyBase"], top_n: int | None = None) -> None:
        if len(opponent_population) == 0:
            self.fitness = -float("inf")
            return

        # print(opponent_population[0].fitness, "fitness of first opponent", top_n)
        # Sort defenders by their current fitness (higher is better)
        sorted_opponents = sorted(opponent_population, key=lambda x: x.fitness, reverse=True)
        assert top_n is not None
        assert len(sorted_opponents) >= top_n, "Not enough opponents to evaluate against."
        top_n_opponents = sorted_opponents[:top_n]

        max_attacker_payoff = -float("inf")

        if len(top_n_opponents) == 0:   # Handle empty or small defender population
            self.fitness = -float('inf')
            return

        for strategy in top_n_opponents:
            combined_policy = CombinedPolicy(
                defender_module=strategy,
                attacker_module=self,
            )
            payoff = combined_policy.single_run(env, 1, add_logs=False)[..., 1].sum().item()

            if payoff > max_attacker_payoff:
                max_attacker_payoff = payoff

        self.fitness = max_attacker_payoff

    def save(self, run_name: str, player_name: str) -> None:
        save_path = self.get_path_name(run_name, player_name)
        torch.save({
            "pure_strategy": self.pure_strategy,
            "fitness": self.fitness,
        }, save_path)

    def load(self, path: str) -> None:
        data = torch.load(path)
        self.pure_strategy = data["pure_strategy"]
        self.fitness = data["fitness"]


class MixedStrategy(StrategyBase):
    def __init__(self, num_nodes: int, pure_strategies: list[PureStrategy], probabilities: list[float]) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.pure_strategies = pure_strategies
        self.probabilities = probabilities
        self.pure_strategy_generator = StrategyGenerator(num_nodes, PureStrategy)

        if len(pure_strategies) != len(probabilities):
            raise ValueError("Number of pure strategies must match number of probabilities.")

        self.sampler = WeightedRandomSampler(
            weights=probabilities,
            num_samples=1,
            replacement=True,
        )
        self.most_probable_strategy = torch.argmax(torch.tensor(probabilities))
        self.fitness = -float("inf")

    def _update_sampler(self) -> None:
        if not self.pure_strategies or not self.probabilities:
            raise ValueError("Invalid pure strategies or probabilities. Cannot update sampler.")
        assert len(self.pure_strategies) == len(self.probabilities)

        self.sampler = WeightedRandomSampler(
            weights=self.probabilities,
            num_samples=1,
            replacement=True,
        )
        # Store original index of most probable
        self.most_probable_strategy_idx = torch.argmax(torch.tensor(self.probabilities)).item()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.pure_strategies or not self.probabilities:
            raise ValueError("Invalid pure strategies or probabilities. Cannot update sampler.")

        if get_interaction_type() == ExplorationType.RANDOM:
            idx = next(iter(self.sampler))
        else:
            idx = self.most_probable_strategy.item()

        try:
            return self.pure_strategies[idx](tensordict)
        except:
            return self.pure_strategies[0](tensordict)

    def sample_pure_strategy(self) -> "PureStrategy":
        if not self.pure_strategies:
            raise ValueError("No pure strategies available to sample from.")
        return self.pure_strategies[torch.randint(0, len(self.pure_strategies), (1,)).item()]

    def mutate(self) -> None:  # Corrected mutation
        if not self.pure_strategies:
            raise ValueError("No pure strategies available to mutate.")

        idx_to_mutate = torch.randint(0, len(self.pure_strategies), (1,)).item()

        # Mutate this component in-place using PureStrategy's mutation logic
        self.pure_strategies[idx_to_mutate].mutate()
        # After mutation, the MixedStrategy might need re-simplification if this
        # component became identical to another, or its fitness evaluation changes.
        # Simplification is typically handled after a batch of mutations/crossovers.

    def crossover(self, other: "StrategyBase") -> tuple["StrategyBase", "StrategyBase"]:
        if not isinstance(other, MixedStrategy):
            raise ValueError("Crossover can only be performed with another MixedStrategy.")

        if not self.pure_strategies and not other.pure_strategies:
            return self.copy(), other.copy()  # Both empty, return copies
        elif not self.pure_strategies:
            # Self is empty, other is not. Children could be copies or one gets all of other's.
            # Let's make child1 a copy of other, child2 a copy of self (empty)
            # This needs careful design if empty strategies are common.
            # For now, let's assume non-empty for typical crossover.
            # If one is empty, the "merge and halve" becomes tricky.
            # A simple approach: if one is empty, its contribution is nothing.
            child1_pure = [ps.copy() for ps in self.pure_strategies] + [ps.copy() for ps in other.pure_strategies]
            child1_probs = [p / 2.0 for p in self.probabilities] + [p / 2.0 for p in other.probabilities]
        else:  # Both have strategies
            child1_pure = [ps.copy() for ps in self.pure_strategies] + [ps.copy() for ps in other.pure_strategies]
            child1_probs = [p / 2.0 for p in self.probabilities] + [p / 2.0 for p in other.probabilities]

        child2_pure = [ps.copy() for ps in other.pure_strategies] + [ps.copy() for ps in self.pure_strategies]
        child2_probs = [p / 2.0 for p in other.probabilities] + [p / 2.0 for p in self.probabilities]

        child1 = MixedStrategy(self.num_nodes, child1_pure, child1_probs)
        child2 = MixedStrategy(self.num_nodes, child2_pure, child2_probs)

        return child1, child2

    def copy(self) -> "MixedStrategy":
        copied_pure_strategies = [pure.copy() for pure in self.pure_strategies]
        copied_probabilities = copy.deepcopy(self.probabilities)
        strategy = MixedStrategy(self.num_nodes, copied_pure_strategies, copied_probabilities)
        strategy.fitness = self.fitness
        return strategy

    def simplify(self, min_prob_threshold: float = 1e-4, max_pure_strategies: int = 50) -> None:
        """ Simplifies the mixed strategy: merges duplicates, removes low probability entries. """
        if len(self.pure_strategies) <= 1:
            self._update_sampler()
            return

        # --- Merge Duplicates ---
        # Convert pure strategies to tuples for hashing
        strategy_map: dict[tuple[int, ...], float] = {}
        for i, pure_strat in enumerate(self.pure_strategies):
            strat_tuple = tuple(pure_strat.pure_strategy.tolist())
            current_prob = strategy_map.get(strat_tuple, 0.0)
            strategy_map[strat_tuple] = current_prob + self.probabilities[i]

        # --- Filter by Probability Threshold ---
        filtered_strategies_tuples: list[tuple[int,...]] = []
        filtered_probabilities: list[float] = []
        for strat_tuple, prob in strategy_map.items():
            if prob >= min_prob_threshold:
                filtered_strategies_tuples.append(strat_tuple)
                filtered_probabilities.append(prob)

        if len(filtered_strategies_tuples) == 0:  # Avoid empty strategy if all below threshold
            # Keep the single highest probability one if possible
            if strategy_map:
                best_strat_tuple, best_prob = max(strategy_map.items(), key=lambda item: item[1])
                self.pure_strategies = [PureStrategy(self.num_nodes, torch.tensor(list(best_strat_tuple), dtype=torch.int32))]
                self.probabilities = [1.0]
            else:  # Should not happen if simplification is called correctly
                self.pure_strategies = []
                self.probabilities = []
            self._update_sampler()
            return

        # --- Limit Number of Pure Strategies (Optional, based on EASG paper's simplification) ---
        if len(filtered_strategies_tuples) > max_pure_strategies:
            # Sort by probability and keep the top ones
            indices = torch.argsort(torch.tensor(filtered_probabilities), descending=True).tolist()[:max_pure_strategies]
            self.pure_strategies = [
                PureStrategy(self.num_nodes, torch.tensor(list(filtered_strategies_tuples[i]), dtype=torch.int32)) for
                i in indices]
            self.probabilities = [filtered_probabilities[i] for i in indices]
        else:
            self.pure_strategies = [PureStrategy(self.num_nodes, torch.tensor(list(s_tuple), dtype=torch.int32)) for
                                    s_tuple in filtered_strategies_tuples]
            self.probabilities = filtered_probabilities

        # --- Renormalize ---
        prob_sum = sum(filtered_probabilities)
        if prob_sum > 0:  # Avoid division by zero if list becomes empty
            self.probabilities = [p / prob_sum for p in self.probabilities]
        else:  # If everything got filtered (unlikely with checks above)
            raise ValueError("It should not happen")

        self._update_sampler()

    def evaluate(self, env: FlipItEnv, opponent_population: list["StrategyBase"], top_n: int | None = None) -> None:
        if len(opponent_population) == 0:
            self.fitness = -float('inf')  # Cannot evaluate
            return

        best_attacker_response_payoff_for_attacker = -float('inf')
        corresponding_defender_payoff = -float('inf')

        for attacker_strategy in opponent_population:
            # Simulate game
            combined_policy = CombinedPolicy(
                defender_module=self,
                attacker_module=attacker_strategy,
            )
            rewards_tensordict = combined_policy.single_run(env, 0, add_logs=False, exploration_type=ExplorationType.RANDOM)
            def_payoff = rewards_tensordict[..., 0].sum().item()
            att_payoff = rewards_tensordict[..., 1].sum().item()

            if att_payoff > best_attacker_response_payoff_for_attacker:
                best_attacker_response_payoff_for_attacker = att_payoff
                corresponding_defender_payoff = def_payoff
                # Tie-breaking: if attacker payoffs are equal, attacker chooses one that is best for defender
                # (Strong Stackelberg Equilibrium assumption from paper)
            elif att_payoff == best_attacker_response_payoff_for_attacker:
                if def_payoff > corresponding_defender_payoff:
                    corresponding_defender_payoff = def_payoff

        # Defender's fitness is their payoff against the attacker's best response (found in pop)
        self.fitness = corresponding_defender_payoff

    def save(self, run_name: str, player_name: str) -> None:
        save_path = self.get_path_name(run_name, player_name)
        torch.save({
            "pure_strategies": torch.stack([strategy.pure_strategy for strategy in self.pure_strategies], dim=0),
            "probabilities": torch.tensor(self.probabilities),
            "fitness": self.fitness,
        }, save_path)

    def load(self, path: str) -> None:
        data = torch.load(path)
        pure_strategies = data["pure_strategies"]
        self.pure_strategies = []
        for strategy in pure_strategies:
            self.pure_strategies.append(PureStrategy(self.num_nodes, strategy))
        self.probabilities = data["probabilities"].tolist()
        self.fitness = data["fitness"]


class StrategyGenerator:
    def __init__(self, num_nodes: int, strategy_class: type[StrategyBase]) -> None:
        self.num_nodes = num_nodes
        self.strategy_class = strategy_class

    def __call__(self, **kwargs) -> StrategyBase:
        return self.strategy_class(self.num_nodes, **kwargs)


# Worker function for multiprocessing (must be defined at the top level or be a static method)
def _worker_evaluate_strategy(strategy_to_eval, env_map, env_num_steps, opponent_population, top_n_val):
    current_env = FlipItEnv(env_map, env_num_steps)

    # Make a deepcopy of the strategy to avoid any potential race conditions
    # This ensures the strategy object in the worker is independent
    strategy_copy = strategy_to_eval.copy()

    strategy_copy.evaluate(current_env, opponent_population, top_n_val)
    return strategy_copy.fitness


class CoevoSGAgentBase(BaseAgent, ABC):
    def __init__(
        self,
        num_nodes: int,
        player_type: int,
        device: torch.device | str,
        run_name: str,
        config: CoevoSGConfig,
        env: FlipItEnv,
        pop_size: int = 200,
        agent_id: int | None = None,
        pool: multiprocessing.pool.Pool | None = None,
    ) -> None:
        super().__init__(num_nodes, player_type, device, run_name, agent_id)

        self.config = config
        self.pop_size = pop_size
        self.env = env
        self.pool = pool

        self.population: list[StrategyBase] = []
        self.generations_no_improvement = 0
        self.current_generation = 0

        self.initialize_population()

    @abstractmethod
    def initialize_population(self) -> None:
        """
        Initialize the population with random pure strategies.
        """

    @property
    def best_population(self) -> StrategyBase:
        assert len(self.population) > 0
        return max(self.population, key=lambda x: x.fitness)

    def save(self) -> None:
        self.best_population.save(self.run_name, self.player_name)

    @abstractmethod
    def load(self, path: str) -> None:
        """ Load the best strategy from disk."""

    def selection(self) -> list[StrategyBase]:
        """ Performs selection using elitism and binary tournament. """
        # Sort by fitness (descending)
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # Elitism
        next_population = [strat.copy() for strat in sorted_population[:self.config.elite_size]]

        # Fill the rest with tournament selection
        while len(next_population) < self.pop_size:
            # Select two individuals randomly
            p1_idx = torch.randint(0, len(self.population), (1,)).item()
            p2_idx = torch.randint(0, len(self.population), (1,)).item()
            p1 = self.population[p1_idx]
            p2 = self.population[p2_idx]

            # Tournament
            winner, loser = (p1, p2) if p1.fitness >= p2.fitness else (p2, p1)

            # Select winner with probability
            if torch.rand((1,)).item() < self.config.selection_pressure:
                next_population.append(winner.copy())
            else:
                next_population.append(loser.copy())

        return next_population

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.best_population(tensordict)

    def evaluate_population(self, opponent_population: list[StrategyBase]) -> None:
        """ Evaluates fitness for all individuals in a specified population. """
        if not opponent_population:
            for strategy in self.population:
                strategy.fitness = -float('inf')
            return

        if self.pool and len(self.population) > 1 and opponent_population:
            worker_with_context = partial(_worker_evaluate_strategy,
                                          env_map=self.env.map,
                                          env_num_steps=self.env.num_steps,
                                          opponent_population=opponent_population,
                                          top_n_val=self.config.attacker_eval_top_n)

            # The iterable for pool.map now only contains the single item that changes per task.
            # This is massively more efficient.
            tasks_iterable = self.population

            results_fitness = self.pool.map(worker_with_context, tasks_iterable)

            for i, fitness_val in enumerate(results_fitness):
                self.population[i].fitness = fitness_val

        else:
            for strategy in self.population:
                strategy.evaluate(self.env, opponent_population, self.config.attacker_eval_top_n)

    def _train_epoch(self) -> None:
        selected_population = self.selection()

        offspring_pool: list[StrategyBase] = []

        next_gen_population = selected_population[:self.config.elite_size]

        # Generate remaining offspring to fill the population
        num_offspring_needed = self.pop_size - len(next_gen_population)

        # Parent pool for crossover/mutation are the selected individuals
        parent_pool = selected_population
        if not parent_pool:  # Should not happen if selected_population is not empty
            self.population = next_gen_population  # Only elites if any
            return

        current_offspring_count = 0
        while current_offspring_count < num_offspring_needed:
            # Select parents for crossover (e.g., from selected_population)
            parent1 = selected_population[torch.randint(0, len(selected_population), torch.Size(())).item()]
            parent2 = selected_population[torch.randint(0, len(selected_population), torch.Size(())).item()]

            if torch.rand((1,)).item() < self.config.crossover_prob:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if torch.rand((1,)).item() < self.config.mutation_prob:
                child1.mutate()
            if torch.rand((1,)).item() < self.config.mutation_prob:
                child2.mutate()

            child1.simplify()
            child2.simplify()

            next_gen_population.append(child1)
            current_offspring_count += 1
            if current_offspring_count < num_offspring_needed:
                next_gen_population.append(child2)
                current_offspring_count += 1
            # print(len(offspring_pool), "offspring created")

        # Combine elites and offspring
        self.population = next_gen_population[:self.pop_size]

    def train_cycle(self, opponent: "CoevoSGAgentBase") -> None:
        # self.evaluate_population(opponent.population)
        # print("population evaluated")
        for _ in range(self.config.gen_per_switch):
            self._train_epoch()
            self.evaluate_population(opponent.population)


class CoevoSGDefenderAgent(CoevoSGAgentBase):
    def initialize_population(self) -> None:
        """
        Initialize the population with random pure strategies.
        """

        self.population = []
        for _ in range(self.pop_size):
            pure_strategy = generate_random_pure_strategy(Player(self.player_type), self.env)
            self.population.append(MixedStrategy(self.num_nodes, [PureStrategy(self.num_nodes, pure_strategy)], [1.0]))

    def load(self, path: str) -> None:
        """
        Load the best strategy from disk.
        """
        self.population = [MixedStrategy(self.num_nodes, [PureStrategy(self.num_nodes, torch.tensor([]))], [1.0])]
        self.best_population.load(path)


class CoevoSGAttackerAgent(CoevoSGAgentBase):
    def initialize_population(self) -> None:
        """
        Initialize the population with random pure strategies.
        """

        self.population = []
        for _ in range(self.pop_size):
            pure_strategy = generate_random_pure_strategy(Player(self.player_type), self.env)
            self.population.append(PureStrategy(self.num_nodes, pure_strategy))

    def load(self, path: str) -> None:
        """
        Load the best strategy from disk.
        """
        self.population = [PureStrategy(self.num_nodes, torch.tensor([]))]
        self.best_population.load(path)
