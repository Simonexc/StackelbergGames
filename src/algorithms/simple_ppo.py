import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim

        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: outputs logits for action probabilities
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head: outputs state value
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor | None = None) -> tuple[Categorical, torch.Tensor]:
        """
        Forward pass.
        Args:
            state: The input state representation.
            action_mask: Boolean tensor indicating valid actions (1=valid, 0=invalid).

        Returns:
            action_dist: A Categorical distribution over actions.
            value: The estimated state value.
        """
        shared_features = self.shared_layer(state)
        action_logits = self.actor_head(shared_features)
        value = self.critic_head(shared_features)

        if action_mask is not None:
            # Apply mask: set logits of invalid actions to -infinity
            # Ensure mask is broadcastable if needed (e.g., state has batch dim)
            if action_mask.dim() < action_logits.dim():
                 action_mask = action_mask.unsqueeze(0).expand_as(action_logits)
            action_logits[~action_mask] = -float('inf')

        action_dist = Categorical(logits=action_logits)
        return action_dist, value

    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor | None = None, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects an action based on the current policy."""
        action_dist, value = self.forward(state, action_mask)
        if deterministic:
             # Handle potential -inf from masking if all actions masked (shouldn't happen with valid mask)
            probs = action_dist.probs
            if torch.all(torch.isinf(probs)): # Or check mask directly
                 # Default/fallback action if needed, though ideally masking prevents this state
                 action = torch.tensor(0, device=state.device) # e.g., observe node 0
            else:
                 action = torch.argmax(probs, dim=-1)
        else:
            action = action_dist.sample()

        log_prob = action_dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor, action_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates the log probability, value, and entropy for given states and actions."""
        action_dist, value = self.forward(state, action_mask)
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        return log_prob, value.squeeze(-1), entropy # Ensure value is squeezed


class PPOBuffer:
    def __init__(self, buffer_size: int, state_dim: int, gamma: float, gae_lambda: float, action_space_size: int, device: torch.device):
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        # Note: Attacker state_dim will be larger if it includes defender policy
        self.actions = torch.zeros(buffer_size, dtype=torch.int64, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        self.action_masks = torch.zeros((buffer_size, action_space_size), dtype=torch.bool, device=device) # Store masks used

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.buffer_size = buffer_size
        self.ptr = 0
        self.path_start_idx = 0
        self.device = device

    def store(self, state, action, reward, value, log_prob, done, action_mask):
        if self.ptr >= self.buffer_size:
            print("Warning: PPO buffer overflow")
            return # Or implement overwrite logic
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.action_masks[self.ptr] = action_mask
        self.ptr += 1

    def finish_path(self, last_value=0.0):
        """Call at the end of an episode or when buffer fills."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice].cpu().numpy(), last_value)
        values = np.append(self.values[path_slice].cpu().numpy(), last_value)
        print(rewards.shape, values.shape)

        # GAE calculation
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - self.dones[path_slice].cpu().numpy()) - values[:-1]
        print(deltas.shape)
        adv = np.zeros_like(rewards)[:-1]
        last_gae_lam = 0
        for t in reversed(range(len(deltas))):
            adv[t] = last_gae_lam = deltas[t] + self.gamma * self.gae_lambda * (1 - self.dones[path_slice][t].cpu().numpy()) * last_gae_lam

        # Advantages and Value Targets (Returns)
        self.advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)
        self.returns = self.advantages + self.values[path_slice] # V_target = A_t + V(s_t)

        self.path_start_idx = self.ptr
        # Ensure ptr doesn't exceed buffer_size for next path
        if self.ptr > self.buffer_size:
            self.ptr = 0
            self.path_start_idx = 0

    def get_batch(self, batch_size: int) -> dict:
        """Returns batches of experiences."""
        if self.ptr < self.buffer_size:
             print(f"Warning: Trying to sample from incomplete buffer ({self.ptr}/{self.buffer_size})")
             # Decide how to handle: return smaller batch, error, or wait?
             # For simplicity, let's work with available data but normalize advantages based on full buffer if possible
             current_data_size = self.ptr
             if current_data_size == 0:
                 return None # No data
        else:
             current_data_size = self.buffer_size


        indices = np.random.permutation(current_data_size)

        # Normalize advantages (optional but recommended)
        adv_mean = self.advantages[:current_data_size].mean()
        adv_std = self.advantages[:current_data_size].std() + 1e-8
        normalized_advantages = (self.advantages[:current_data_size] - adv_mean) / adv_std

        start_idx = 0
        while start_idx < current_data_size:
            batch_indices = indices[start_idx : min(start_idx + batch_size, current_data_size)]
            print(batch_indices, normalized_advantages.shape, current_data_size, self.buffer_size, self.advantages)
            yield {
                'states': self.states[batch_indices],
                'actions': self.actions[batch_indices],
                'log_probs_old': self.log_probs[batch_indices],
                'returns': self.returns[batch_indices],
                'advantages': normalized_advantages[batch_indices],
                'action_masks': self.action_masks[batch_indices]
            }
            start_idx += batch_size

    def clear(self):
        self.ptr = 0
        self.path_start_idx = 0
        # Optionally clear tensors if memory is a concern, otherwise they'll be overwritten
