import torch
import torch.optim as optim
from tqdm import tqdm

from config import Player
from environments.flipit import FlipItEnv, FlipItMap
from environments.flipit_utils import Action, ActionTargetPair, PlayerTargetPair
from algorithms.simple_ppo import ActorCritic, PPOBuffer


def get_state_representation(env: FlipItEnv, device: torch.device) -> torch.Tensor:
    """Creates a tensor representation of the current state."""
    owners_tensor = env.node_owners.float().to(device)
    # Normalize step count
    step_tensor = torch.tensor([env.current_step / env.m], dtype=torch.float32, device=device)
    return torch.cat((owners_tensor, step_tensor))


def get_action_mask(env: FlipItEnv, player: Player, device: torch.device) -> torch.Tensor:
    """Generates a boolean mask for valid actions."""
    n = env.n
    mask = torch.zeros(env.action_space.shape[0], dtype=torch.bool, device=device) # Size = 2 * n

    # Observe actions are always valid for existing nodes
    mask[0:n] = True

    # Flip actions (n to 2n-1) require reachability
    for target_node in range(n):
        try:
            # Use internal reachability check - might need adjustment if env class changes
            if env._is_reachable(player, target_node):
                 # Map node index to action index (n + node_index)
                 action_index = n + target_node
                 mask[action_index] = True
        except ValueError: # Node doesn't exist (shouldn't happen with 0..n-1)
            pass
        except AttributeError: # Handle case where env._is_reachable might not exist
            print("Warning: Cannot check reachability, assuming all flips possible.")
            mask[n:] = True # Fallback: allow all flips if check fails

    # Ensure at least one action is valid (e.g., if all flips are somehow invalid)
    if not mask.any():
        mask[0] = True # Default to observing node 0 if nothing else is valid

    return mask


def map_action_index_to_pair(action_index: int, n: int) -> ActionTargetPair:
    """Converts flat action index back to ActionTargetPair."""
    if action_index < n:
        return ActionTargetPair(action=Action.observe, target_node=action_index)
    else:
        return ActionTargetPair(action=Action.flip, target_node=action_index - n)


def train_flipit_ppo(
    env: FlipItEnv,
    num_iterations: int = 100,
    defender_epochs_per_iter: int = 5,
    attacker_epochs_per_iter: int = 5,
    ppo_update_epochs: int = 10,
    num_episodes_per_epoch: int = 20, # Num episodes to collect data for each agent's training epoch
    batch_size: int = 64,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    vf_coef: float = 0.5, # Value function loss coefficient
    entropy_coef: float = 0.01, # Entropy bonus coefficient
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Trains Defender and Attacker agents iteratively using PPO.
    """
    n = env.n
    m = env.m
    action_dim = env.action_space.shape[0] # 2 * n

    # State dimension for Defender
    defender_state_dim = n + 1 # node_owners + normalized_step

    # State dimension for Attacker (includes Defender's policy distribution)
    attacker_state_dim = n + 1 + action_dim # node_owners + normalized_step + def_policy_probs

    # Initialize Actor-Critic networks
    defender_ac = ActorCritic(defender_state_dim, action_dim, hidden_dim).to(device)
    attacker_ac = ActorCritic(attacker_state_dim, action_dim, hidden_dim).to(device)

    # Initialize Optimizers
    defender_optimizer = optim.Adam(defender_ac.parameters(), lr=lr)
    attacker_optimizer = optim.Adam(attacker_ac.parameters(), lr=lr)

    # Initialize PPO Buffers
    # Buffer size should be large enough to hold data from num_episodes_per_epoch * m steps
    buffer_size = num_episodes_per_epoch * m
    defender_buffer = PPOBuffer(buffer_size, defender_state_dim, gamma, gae_lambda, env.action_space.shape[0], device)
    # Attacker buffer needs space for the larger state representation
    attacker_buffer = PPOBuffer(buffer_size, attacker_state_dim, gamma, gae_lambda, env.action_space.shape[0], device)

    print(f"Starting PPO training on device: {device}")
    print(f"Action space size: {action_dim}")
    print(f"Defender state dim: {defender_state_dim}")
    print(f"Attacker state dim: {attacker_state_dim}")

    # --- Main Training Loop ---
    for iteration in range(num_iterations):
        print(f"\n===== Iteration {iteration+1}/{num_iterations} =====")

        # --- Train Defender ---
        print("--- Training Defender ---")
        defender_ac.train()
        attacker_ac.eval() # Keep attacker fixed while training defender

        for epoch in range(defender_epochs_per_iter):
            defender_buffer.clear()
            avg_def_reward_epoch = 0

            # Collect experience
            for _ in range(num_episodes_per_epoch):
                env.reset()
                episode_def_reward = 0
                last_def_value = 0 # For GAE calculation at episode end

                for t in range(m):
                    # Get current state representations
                    state_def = get_state_representation(env, device)
                    state_att_base = get_state_representation(env, device) # Base state for attacker

                    # Get action masks
                    mask_def = get_action_mask(env, Player.defender, device)
                    mask_att = get_action_mask(env, Player.attacker, device)

                    with torch.no_grad():
                        # Defender action
                        action_def_idx, log_prob_def, value_def = defender_ac.get_action(state_def, mask_def)

                        # Attacker action (needs fixed defender policy for its input)
                        def_policy_dist, _ = defender_ac(state_def, mask_def) # Get current defender policy dist
                        attacker_input_state = torch.cat((state_att_base, def_policy_dist.probs.detach()), dim=-1)
                        action_att_idx, _, _ = attacker_ac.get_action(attacker_input_state, mask_att) # Attacker acts based on its policy

                    # Environment step
                    action_def_pair = map_action_index_to_pair(action_def_idx.item(), n)
                    action_att_pair = map_action_index_to_pair(action_att_idx.item(), n)
                    _, rewards, _, done = env.step(action_def_pair, action_att_pair) # Ignoring observe_results for now

                    # Store Defender's transition
                    reward_def = rewards[Player.defender.value]
                    defender_buffer.store(state_def, action_def_idx, reward_def, value_def.item(), log_prob_def, done, mask_def)
                    episode_def_reward += reward_def
                    print(defender_buffer.ptr)
                    if done:
                        last_def_value = 0.0 # Terminal state value is 0
                        break
                    elif t == m - 1: # End of episode, get value of final state
                         with torch.no_grad():
                             next_state_def = get_state_representation(env, device)
                             _, last_def_value_tensor = defender_ac(next_state_def, get_action_mask(env, Player.defender, device))
                             last_def_value = last_def_value_tensor.item()


                avg_def_reward_epoch += episode_def_reward
                defender_buffer.finish_path(last_def_value) # Compute advantages for the episode

            avg_def_reward_epoch /= num_episodes_per_epoch
            print(f"Defender Epoch {epoch+1}/{defender_epochs_per_iter}, Avg Reward: {avg_def_reward_epoch:.2f}")


            # Perform PPO updates for Defender
            defender_ac.train() # Ensure model is in training mode for updates
            for _ in range(ppo_update_epochs):
                batch_generator = defender_buffer.get_batch(batch_size)
                if batch_generator is None: continue # Skip if buffer empty

                for batch in batch_generator:
                    states = batch['states']
                    actions = batch['actions']
                    log_probs_old = batch['log_probs_old']
                    returns = batch['returns']
                    advantages = batch['advantages']
                    action_masks = batch['action_masks']

                    # Evaluate current policy on batch
                    new_log_probs, values, entropy = defender_ac.evaluate_actions(states, actions, action_masks)

                    # PPO Ratio
                    ratio = torch.exp(new_log_probs - log_probs_old)

                    # Clipped Surrogate Objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Function Loss (MSE)
                    value_loss = F.mse_loss(values, returns)

                    # Total Loss
                    loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy.mean()

                    # Gradient descent step
                    defender_optimizer.zero_grad()
                    loss.backward()
                    # Optional: Gradient clipping
                    # torch.nn.utils.clip_grad_norm_(defender_ac.parameters(), max_norm=0.5)
                    defender_optimizer.step()


        # --- Train Attacker ---
        print("--- Training Attacker ---")
        attacker_ac.train()
        defender_ac.eval() # Keep defender fixed

        for epoch in range(attacker_epochs_per_iter):
            attacker_buffer.clear()
            avg_att_reward_epoch = 0

            # Collect experience
            for _ in range(num_episodes_per_epoch):
                env.reset()
                episode_att_reward = 0
                last_att_value = 0.0

                for t in range(m):
                    # Get state representations
                    state_def = get_state_representation(env, device) # Needed for defender policy
                    state_att_base = get_state_representation(env, device)

                    # Get action masks
                    mask_def = get_action_mask(env, Player.defender, device)
                    mask_att = get_action_mask(env, Player.attacker, device)

                    with torch.no_grad():
                        # Defender action (using its fixed current policy)
                        def_policy_dist, _ = defender_ac(state_def, mask_def)
                        action_def_idx, _, _ = defender_ac.get_action(state_def, mask_def)

                        # Attacker action (input includes defender policy)
                        attacker_input_state = torch.cat((state_att_base, def_policy_dist.probs.detach()), dim=-1)
                        action_att_idx, log_prob_att, value_att = attacker_ac.get_action(attacker_input_state, mask_att)

                    # Environment step
                    action_def_pair = map_action_index_to_pair(action_def_idx.item(), n)
                    action_att_pair = map_action_index_to_pair(action_att_idx.item(), n)
                    _, rewards, _, done = env.step(action_def_pair, action_att_pair)

                    # Store Attacker's transition
                    reward_att = rewards[Player.attacker.value]
                    attacker_buffer.store(attacker_input_state, action_att_idx, reward_att, value_att.item(), log_prob_att, done, mask_att)
                    episode_att_reward += reward_att

                    if done:
                        last_att_value = 0.0
                        break
                    elif t == m - 1:
                         with torch.no_grad():
                             next_state_def = get_state_representation(env, device)
                             next_state_att_base = get_state_representation(env, device)
                             next_def_policy_dist, _ = defender_ac(next_state_def, get_action_mask(env, Player.defender, device))
                             next_attacker_input_state = torch.cat((next_state_att_base, next_def_policy_dist.probs.detach()), dim=-1)
                             _, last_att_value_tensor = attacker_ac(next_attacker_input_state, get_action_mask(env, Player.attacker, device))
                             last_att_value = last_att_value_tensor.item()


                avg_att_reward_epoch += episode_att_reward
                attacker_buffer.finish_path(last_att_value) # Compute advantages

            avg_att_reward_epoch /= num_episodes_per_epoch
            print(f"Attacker Epoch {epoch+1}/{attacker_epochs_per_iter}, Avg Reward: {avg_att_reward_epoch:.2f}")


            # Perform PPO updates for Attacker
            attacker_ac.train()
            for _ in range(ppo_update_epochs):
                 batch_generator = attacker_buffer.get_batch(batch_size)
                 if batch_generator is None: continue

                 for batch in batch_generator:
                    states = batch['states'] # Note: These states include the defender policy dist
                    actions = batch['actions']
                    log_probs_old = batch['log_probs_old']
                    returns = batch['returns']
                    advantages = batch['advantages']
                    action_masks = batch['action_masks']


                    # Evaluate current policy
                    new_log_probs, values, entropy = attacker_ac.evaluate_actions(states, actions, action_masks)

                    # PPO Ratio
                    ratio = torch.exp(new_log_probs - log_probs_old)

                    # Clipped Surrogate Objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Function Loss
                    value_loss = F.mse_loss(values, returns)

                    # Total Loss
                    loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy.mean()

                    # Gradient descent step
                    attacker_optimizer.zero_grad()
                    loss.backward()
                    # Optional: Gradient clipping
                    # torch.nn.utils.clip_grad_norm_(attacker_ac.parameters(), max_norm=0.5)
                    attacker_optimizer.step()

        # --- End of Iteration ---
        # Optional: Save models, evaluate against each other or fixed policies, log metrics


    print("Training finished.")
    return defender_ac, attacker_ac


# --- Example Usage ---
if __name__ == "__main__":
    flipit_map = FlipItMap.load("test.pth")
    num_steps = 5
    env = FlipItEnv.from_map(num_steps=num_steps, flipit_map=flipit_map)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    defender_policy, attacker_policy = train_flipit_ppo(
        env,
        num_iterations=1, # Adjust as needed
        defender_epochs_per_iter=2,
        attacker_epochs_per_iter=2,
        ppo_update_epochs=4,
        num_episodes_per_epoch=32, # Collect more data per update cycle
        batch_size=128, # Larger batch size for stability
        hidden_dim=128,
        lr=1e-4, # May need tuning
        device=device
    )

    # Save models if needed
    torch.save(defender_policy.state_dict(), "defender_policy.pth")
    torch.save(attacker_policy.state_dict(), "attacker_policy.pth")
