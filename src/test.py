from environments.flipit import FlipItEnv
from algorithms.coevosg import CoEvoSG


NUM_NODES = 10  # n
NUM_STEPS = 5   # m
SEED = 42

flipit_game = FlipItEnv(num_nodes=NUM_NODES, num_steps=NUM_STEPS, seed=SEED)

# --- CoEvoSG Setup & Training ---
coevosg_solver = CoEvoSG(
    env=flipit_game,
    defender_pop_size=50,   # Smaller pops for quicker demo
    attacker_pop_size=50,
    generations=100,        # Fewer generations for demo
    gen_per_switch=5,
    elite_size=2,
    crossover_prob=0.8,
    mutation_prob=0.5,
    selection_pressure=0.9,
    attacker_eval_top_n=5,
    no_improvement_limit=20, # Stop if no improvement for 20*gp generations
)

best_strategy = coevosg_solver.train()

# --- Analyze the result ---
if best_strategy:
    print("\n--- Best Defender Strategy Found ---")
    print(f"Fitness (Expected Payoff): {best_strategy.fitness:.4f}")
    print(f"Number of pure strategies in mix: {len(best_strategy.pure_strategies)}")
    print("Probabilities:", [f"{p:.3f}" for p in best_strategy.probabilities])
    # Optionally print one of the pure strategies
    if best_strategy.pure_strategies:
        print("\nExample Pure Strategy (m steps):")
        for step, action_pair in enumerate(best_strategy.pure_strategies[0]):
             print(f"  Step {step+1}: Action={action_pair.action.name}, Target={action_pair.target_node}")

else:
    print("\nNo best strategy found (training might have failed or population emptied).")