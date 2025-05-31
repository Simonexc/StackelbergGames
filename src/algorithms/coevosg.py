import copy
import random
import time

from tqdm import tqdm

from environments.flipit import FlipItEnv
from environments.flipit_utils import generate_random_pure_strategy, run_simulation, Action, ActionTargetPair

from strategies import MixedStrategy, PureStrategy, Policy
from config import Player


class CoEvoSG:
    def __init__(
        self,
        env: FlipItEnv,
        defender_pop_size: int = 200,
        attacker_pop_size: int = 200,
        generations: int = 1000,          # lg in paper (total max generations for Defender)
        gen_per_switch: int = 20,         # gp in paper
        elite_size: int = 2,              # e in paper
        crossover_prob: float = 0.8,      # pc in paper
        mutation_prob: float = 0.5,       # pm in paper
        selection_pressure: float = 0.9,  # ps in paper
        attacker_eval_top_n: int = 10,    # Ntop in paper
        no_improvement_limit: int = 20,   # le in paper
    ) -> None:
        self.env = env
        self.n = env.n
        self.m = env.m
        self.ND = defender_pop_size
        self.NA = attacker_pop_size
        self.lg = generations
        self.gp = gen_per_switch
        self.e = elite_size
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.ps = selection_pressure
        self.Ntop = attacker_eval_top_n
        self.le = no_improvement_limit

        self.defender_population: list[MixedStrategy] = []
        self.attacker_population: list[PureStrategy] = []

        self.best_defender_overall: MixedStrategy | None = None
        self.generations_no_improvement = 0
        self.current_generation = 0 # Tracks defender generations

    def initialize_populations(self) -> None:
        """ Initializes both populations with random strategies. """

        # Attacker: List of pure strategies
        self.attacker_population = []
        for _ in range(self.NA):
            pure_strat = generate_random_pure_strategy(Player.attacker, self.env)
            self.attacker_population.append(PureStrategy(pure_strat))

        # Defender: Start with pure strategies (prob=1.0) as per EASG basis
        self.defender_population = []
        for _ in range(self.ND):
            pure_strat = generate_random_pure_strategy(Player.defender, self.env)
            # Each initial individual has one pure strategy with probability 1.0
            self.defender_population.append(MixedStrategy([pure_strat], [1.0]))

    def evaluate_defender(self, defender_strategy: MixedStrategy) -> None:
        """ Calculates fitness for a Defender individual. """
        best_attacker_response_payoff = -float('inf')
        corresponding_defender_payoff = -float('inf')

        if len(self.attacker_population) == 0:
            defender_strategy.fitness = -float('inf')  # Cannot evaluate
            return

        for attacker_strategy in self.attacker_population:
            # Simulate game
            def_payoff, att_payoff = run_simulation(defender_strategy, attacker_strategy, self.env)

            # Attacker seeks to maximize their own payoff
            if att_payoff > best_attacker_response_payoff:
                best_attacker_response_payoff = att_payoff
                corresponding_defender_payoff = def_payoff

        # Defender's fitness is their payoff against the attacker's best response (found in pop)
        defender_strategy.fitness = corresponding_defender_payoff

    def evaluate_attacker(self, attacker_strategy: PureStrategy) -> None:
        """ Calculates fitness for an Attacker individual. """
        if len(self.defender_population) == 0:
            attacker_strategy.fitness = -float('inf')
            return

        # Sort defenders by their current fitness (higher is better)
        sorted_defenders = sorted(self.defender_population, key=lambda d: d.fitness, reverse=True)
        top_n_defenders = sorted_defenders[:self.Ntop]

        max_attacker_payoff = -float('inf')

        if len(top_n_defenders) == 0:  # Handle empty or small defender population
            attacker_strategy.fitness = -float('inf')
            return

        for defender_strategy in top_n_defenders:
            # Simulate game
            _, att_payoff = run_simulation(defender_strategy, attacker_strategy, self.env)
            # Attacker's fitness is the max payoff they achieve against the top N defenders
            if att_payoff > max_attacker_payoff:
                max_attacker_payoff = att_payoff

        attacker_strategy.fitness = max_attacker_payoff

    def evaluate_population(self, population_type: Player):
        """ Evaluates fitness for all individuals in a specified population. """
        if population_type == Player.defender:
            for defender_strategy in self.defender_population:
                self.evaluate_defender(defender_strategy)
        elif population_type == Player.attacker:
            for attacker_strategy in self.attacker_population:
                self.evaluate_attacker(attacker_strategy)
        else:
            raise ValueError("Invalid population type")

    def crossover_attacker(
        self, parent1: PureStrategy, parent2: PureStrategy
    ) -> tuple[PureStrategy, PureStrategy]:
        strat1 = parent1.pure_strategy
        strat2 = parent2.pure_strategy
        """ One-point crossover for attacker pure strategies. """
        if len(strat1) != self.m or len(strat2) != self.m:
            raise ValueError("Strategy lengths differ from expected m")

        if self.m < 2:  # Cannot crossover length 1
            return PureStrategy(copy.deepcopy(strat1)), PureStrategy(copy.deepcopy(strat2))

        # Simple one-point crossover
        point = random.randint(1, self.m - 1)
        child1_strat = strat1[:point] + strat2[point:]
        child2_strat = strat2[:point] + strat1[point:]

        return PureStrategy(child1_strat), PureStrategy(child2_strat)

    def crossover_defender(
        self, parent1: MixedStrategy, parent2: MixedStrategy
    ) -> tuple[MixedStrategy, MixedStrategy]:
        """ Merges pure strategies and averages probabilities. """
        # Combine pure strategies and halve probabilities
        child1_pure = parent1.pure_strategies + parent2.pure_strategies
        child1_probs = [p / 2.0 for p in parent1.probabilities] + [p / 2.0 for p in parent2.probabilities]

        # Create child (simplification/normalization happens later)
        # For simplicity, we create one child and return it twice (less standard, but avoids complex second child logic)
        # Or, better: Let selection handle picking parents again for the second slot. We generate ONE child here.
        child1 = MixedStrategy(child1_pure, child1_probs)
        # child1.simplify() # Apply simplification immediately or after mutation/selection phase

        # Create a second child (could be identical or use different logic if needed)
        # For now, let's make it similar (could also just return one and let selection fill)
        child2_pure = parent2.pure_strategies + parent1.pure_strategies
        child2_probs = [p / 2.0 for p in parent2.probabilities] + [p / 2.0 for p in parent1.probabilities]
        child2 = MixedStrategy(child2_pure, child2_probs)

        return child1, child2  # Caller should simplify these

    def mutate(self, individual: Policy) -> None:
        pure_strategy = individual.sample_pure_strategy()

        mutation_point = random.randint(0, self.m - 1)
        for i in range(mutation_point, self.m):
            new_action = Action.random()
            new_target = random.randint(0, self.n - 1)

            pure_strategy[i] = ActionTargetPair(new_action, new_target)

    def selection(self, population: list[Strategy], population_size: int) -> list[Strategy]:
        """ Performs selection using elitism and binary tournament. """
        # Sort by fitness (descending)
        sorted_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        # Elitism
        next_population = sorted_population[:self.e]

        # Fill the rest with tournament selection
        while len(next_population) < population_size:
            # Select two individuals randomly
            p1_idx = random.randint(0, len(population) - 1)
            p2_idx = random.randint(0, len(population) - 1)
            p1 = population[p1_idx]
            p2 = population[p2_idx]

            # Tournament
            winner = p1
            loser = p2
            if p2.fitness > p1.fitness:
                winner = p2
                loser = p1

            # Select winner with probability ps
            if random.random() < self.ps:
                next_population.append(copy.deepcopy(winner))  # Add a copy
            else:
                next_population.append(copy.deepcopy(loser))  # Add a copy

        return next_population

    # --- Training Loop ---
    def evolve_generation(self, population_type: Player):
        """ Evolves one generation for the specified population. """
        if population_type == Player.defender:
            population = self.defender_population
            pop_size = self.ND
            crossover_func = self.crossover_defender
        else:  # Attacker
            population = self.attacker_population
            pop_size = self.NA
            crossover_func = self.crossover_attacker

        # 1. Selection
        selected_population = self.selection(population, pop_size)

        # 2. Crossover & Mutation (operate on copies of selected individuals)
        next_gen_population = selected_population[:self.e]  # Keep elites unchanged for now
        offspring_pool = []

        while len(offspring_pool) < pop_size - self.e:
            # Select parents for crossover (e.g., from selected_population)
            parent1 = random.choice(selected_population)
            parent2 = random.choice(selected_population)

            if random.random() < self.pc:
                child1, child2 = crossover_func(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)  # Pass through copies

            # Mutate children
            if random.random() < self.pm:
                self.mutate(child1)
            if random.random() < self.pm:
                self.mutate(child2)

            offspring_pool.append(child1)
            if len(offspring_pool) < pop_size - self.e:
                offspring_pool.append(child2)

        # Combine elites and offspring
        final_next_gen = selected_population[:self.e] + offspring_pool[:pop_size - self.e]

        # Apply Defender Simplification AFTER mutation/crossover potentially created complex ones
        if population_type == Player.defender:
            self.defender_population = []
            for ind in final_next_gen:
                assert isinstance(ind, MixedStrategy), "Expected MixedStrategy for defender"
                ind.simplify()
                if len(ind.pure_strategies) > 0:
                    self.defender_population.append(ind)
        else:
            self.attacker_population = final_next_gen

    def train(self):
        """ Runs the main coevolutionary training loop. """
        self.initialize_populations()

        start_time = time.time()
        defender_generations_total = 0

        # Initial evaluation
        print("Initial evaluation...")
        self.evaluate_population(Player.attacker)
        self.evaluate_population(Player.defender)
        best_defender_current_run = max(self.defender_population, key=lambda d: d.fitness, default=None)
        self.best_defender_overall = copy.deepcopy(best_defender_current_run)
        print(
            f"Initial Best Defender Fitness: {self.best_defender_overall.fitness if self.best_defender_overall else 'N/A'}")

        pbar = tqdm(total=self.lg, desc="Defender Generations")
        while defender_generations_total < self.lg:
            # --- Attacker Evolution Phase ---
            # print(f"\n--- Attacker Evolution Phase (Defender Gen {defender_generations_total+1}) ---")
            for _ in range(self.gp):
                best_attacker_current_run = max(self.attacker_population, key=lambda d: d.fitness, default=None)
                print(best_attacker_current_run.pure_strategy, best_attacker_current_run.fitness)
                # Attacker evolves based on current Defender pop
                self.evolve_generation(Player.attacker)
                # Re-evaluate Attacker fitness against the *same* (fixed during attacker phase) Defender pop
                self.evaluate_population(Player.attacker)
                # Optional: Log attacker fitness progress

            # --- Defender Evolution Phase ---
            # print(f"\n--- Defender Evolution Phase (Defender Gen {defender_generations_total+1}) ---")
            for i in range(self.gp):
                # Defender evolves based on the now-updated Attacker pop
                self.evolve_generation(Player.defender)
                # Re-evaluate Defender fitness against the updated Attacker pop
                self.evaluate_population(Player.defender)

                defender_generations_total += 1
                pbar.update(1)
                if defender_generations_total >= self.lg:
                    break  # Exit if max generations reached mid-phase

            # Update overall best defender
            current_best_in_pop = max(self.defender_population, key=lambda d: d.fitness, default=None)
            if current_best_in_pop and (
                    self.best_defender_overall is None or current_best_in_pop.fitness > self.best_defender_overall.fitness):
                self.best_defender_overall = copy.deepcopy(current_best_in_pop)
                self.generations_no_improvement = 0
                tqdm.write(
                    f"Gen {defender_generations_total}: New best defender fitness = {self.best_defender_overall.fitness:.4f}")
            else:
                self.generations_no_improvement += self.gp  # Increment by number of defender gens in the phase

            # Check stop conditions
            if self.generations_no_improvement >= self.le:
                print(f"\nStopping early: No improvement for {self.le} defender generations.")
                break
            if defender_generations_total >= self.lg:
                print(f"\nStopping: Reached maximum {self.lg} defender generations.")
                break

        pbar.close()
        end_time = time.time()
        print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")
        print(
            f"Best Defender Strategy Fitness: {self.best_defender_overall.fitness if self.best_defender_overall else 'N/A'}")

        return self.best_defender_overall
