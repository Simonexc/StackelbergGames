import os
import random
from typing import TYPE_CHECKING

from tensordict.nn.probabilistic import InteractionType as ExplorationType
import torch
from tqdm import tqdm
from torch import nn
from torchrl.collectors import MultiSyncDataCollector, SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, PrioritizedSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase, ParallelEnv

from config import TrainingConfig
from algorithms.generic_policy import CombinedPolicy, MultiAgentPolicy, BaseAgent, GreedyOracleAgent, RandomAgent

if TYPE_CHECKING:
    from environments.base_env import EnvironmentBase


def create_replay_buffer(config: TrainingConfig) -> ReplayBuffer:
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=config.steps_per_batch),
        sampler=PrioritizedSampler(max_capacity=config.steps_per_batch, alpha=0.7, beta=0.5),#SamplerWithoutReplacement(),
        priority_key="priority",
        # transform=lambda data: data.reshape(-1),
    )


def create_collector(
    env: "EnvironmentBase",
    policy: nn.Module,
    config: TrainingConfig,
    device: torch.device | str,
    num_environments: int,
) -> MultiSyncDataCollector:
    return MultiSyncDataCollector(
        [env.create_from_self for _ in range(num_environments)],
        #ParallelEnv(num_environments, env.create_from_self),
        #env.create_from_self,
        policy=policy,
        frames_per_batch=config.steps_per_batch,
        total_frames=config.total_steps_per_turn,
        split_trajs=False,
        device=device,
        policy_device="cuda:0",
        update_at_each_batch=True,
        exploration_type=ExplorationType.RANDOM,
    )


def train_agent(
    combined_policy: CombinedPolicy,
    env: "EnvironmentBase",
    player: int,
    training_config: TrainingConfig,
    num_envs: int,
    pbar: tqdm,
) -> None:
    collector = create_collector(env, combined_policy, training_config, env.device, num_envs)
    replay_buffer = create_replay_buffer(training_config)

    trained_agent = combined_policy.attacker_module if player == 1 else combined_policy.defender_module
    if player == 0:
        combined_policy.defender_module.train()
        combined_policy.attacker_module.eval()
        combined_policy.defender_module.currently_training = True
        combined_policy.attacker_module.currently_training = False
    else:
        combined_policy.defender_module.eval()
        combined_policy.attacker_module.train()
        combined_policy.defender_module.currently_training = False
        combined_policy.attacker_module.currently_training = True

    for i, tensordict_data in enumerate(collector):
        num_data = tensordict_data.numel()
        reward = trained_agent.train_cycle(tensordict_data, replay_buffer, i)

        pbar.update(num_data)
        cum_reward_str = (
            f"average reward {'defender' if player == 0 else 'attacker'}={reward: 4.4f}"
        )
        pbar.set_description(cum_reward_str)

        # Eval
        combined_policy.evaluate(env, 1000, player)
        combined_policy.single_run(env, player)
        # if i >= 20:
        #     combined_policy.exploration_coeff *= 0.3
        # if combined_policy.exploration_coeff < 0.01:
        #     combined_policy.exploration_coeff = 0.0

        # schedulers[currently_training].step()
    collector.shutdown()
    del collector

    trained_agent.save()


def train_stage(
    combined_policy: CombinedPolicy,
    env: "EnvironmentBase",
    training_configs: tuple[TrainingConfig, TrainingConfig],
    num_envs: int,
    pbar: tqdm,
) -> None:
    for player in range(2):
        training_config = training_configs[player]
        if player == 1 and isinstance(combined_policy.attacker_module, MultiAgentPolicy):
            combined_policy.attacker_module.add_policy()
        train_agent(
            combined_policy,
            env,
            player=player,
            training_config=training_config,
            num_envs=num_envs,
            pbar=pbar,
        )


def train_stage_coevosg(
    combined_policy: CombinedPolicy,
    pbar: tqdm,
) -> None:
    combined_policy.defender_module.train_cycle(combined_policy.attacker_module)
    combined_policy.evaluate(combined_policy.defender_module.env, 100, 0)
    combined_policy.single_run(combined_policy.defender_module.env, 0)
    combined_policy.attacker_module.train_cycle(combined_policy.defender_module)
    combined_policy.evaluate(combined_policy.attacker_module.env, 100, 1)
    combined_policy.single_run(combined_policy.attacker_module.env, 1)

    pbar.update(1)


def compare_agent_pairs(agent_pairs: list[tuple[BaseAgent, BaseAgent, str]], attacker_agents: list[tuple[BaseAgent, str]], env: EnvBase, print_results: bool = True) -> dict[str, float | None]:
    """
    Compare each defender against all attackers using CombinedPolicy.single_run.
    If a policy is MultiAgentPolicy, compare against all its agents and average the result.
    If policy is random or greedy, compare multiple times and average the results.
    agent_pairs: list of (defender, attacker, description) tuples
    env: environment to use for evaluation
    num_runs: number of single_run evaluations to average
    print_results: whether to print the results
    Returns: results as a list of dicts
    """
    results = {}
    all_defender_agents = [(pair[0], pair[2]) for pair in agent_pairs]
    all_attacker_agents = [(pair[1], pair[2]) for pair in agent_pairs] + attacker_agents
    repeat_experiments = 10
    seed_list = [random.randint(0, 10000000) for _ in range(repeat_experiments * repeat_experiments)]
    for i, (defender, def_desc) in enumerate(all_defender_agents):

        defender_rewards = []
        for j, (attacker, att_desc) in enumerate(all_attacker_agents):
            def_agents = [defender]

            if isinstance(attacker, MultiAgentPolicy):
                att_agents = attacker.policies * repeat_experiments
            else:
                att_agents = [attacker] * repeat_experiments * repeat_experiments

            def_rewards, att_rewards = [], []
            for def_agent in def_agents:
                for att_agent, seed in zip(att_agents, seed_list):
                    combined = CombinedPolicy(def_agent, att_agent)
                    env.set_seed(seed)
                    reward = combined.single_run(env, current_player=0, add_logs=False)
                    def_rewards.append(reward[..., 0].sum().item())
                    att_rewards.append(reward[..., 1].sum().item())
                    defender_rewards.append(reward[..., 0].sum().item())

            result = {
                f'{def_desc}/{att_desc}/avg': torch.tensor(def_rewards).mean().item(),
                f"{def_desc}/{att_desc}/std": torch.tensor(def_rewards).std().item() if len(att_agents) > 1 else None,
                f"{def_desc}/{att_desc}/num_samples": len(def_rewards),
                f"{def_desc}/{att_desc}/raw": torch.tensor(def_rewards).tolist(),
            }
            results.update(result)

            if print_results:
                out_string = f"Defender: {def_desc} vs Attacker: {att_desc} => Defender avg reward: {result[f'{def_desc}/{att_desc}/avg']:.4f}"
                if result[f"{def_desc}/{att_desc}/std"] is not None:
                    out_string += f" ({result[f"{def_desc}/{att_desc}/std"]:.4f})"
                print(out_string)
        result = {
            f'{def_desc}/avg': torch.tensor(defender_rewards).mean().item(),
            f'{def_desc}/std': torch.tensor(defender_rewards).std().item() if len(defender_rewards) > 1 else None,
            f'{def_desc}/num_samples': len(defender_rewards),
        }
        results.update(result)
        if print_results:
            out_string = f"Defender: {def_desc} => Avg reward: {result[f'{def_desc}/avg']:.4f}"
            if result[f'{def_desc}/std'] is not None:
                out_string += f" ({result[f'{def_desc}/std']:.4f})"
            print(out_string)

    return results
