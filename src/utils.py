import os

from tensordict.nn.probabilistic import InteractionType as ExplorationType
import torch
from tqdm import tqdm
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import EnvBase

from config import TrainingConfig
from algorithms.generic_policy import CombinedPolicy, MultiAgentPolicy


def create_replay_buffer(config: TrainingConfig) -> ReplayBuffer:
    return ReplayBuffer(
        storage=LazyTensorStorage(max_size=config.steps_per_batch),
        sampler=SamplerWithoutReplacement(),
        transform=lambda data: data.reshape(-1).cpu(),
    )


def create_collector(
    env: EnvBase,
    policy: nn.Module,
    config: TrainingConfig,
    device: torch.device | str,
    num_environments: int,
) -> MultiaSyncDataCollector:
    return MultiaSyncDataCollector(
        [lambda: env for _ in range(num_environments)],
        policy,
        frames_per_batch=config.steps_per_batch,
        total_frames=config.total_steps_per_turn,
        split_trajs=False,
        device=device,
        update_at_each_batch=True,
        exploration_type=ExplorationType.RANDOM,
    )


def train_agent(
    combined_policy: CombinedPolicy,
    env: EnvBase,
    player: int,
    training_config: TrainingConfig,
    device: torch.device | str,
    num_envs: int,
    pbar: tqdm,
) -> None:
    collector = create_collector(env, combined_policy, training_config, device, num_envs)
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
        reward = trained_agent.train_cycle(tensordict_data, replay_buffer, i)

        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward {'defender' if player == 0 else 'attacker'}={reward: 4.4f}"
        )
        pbar.set_description(cum_reward_str)

        # Eval
        combined_policy.evaluate(env, 100, player)
        combined_policy.single_run(env, player)

        # schedulers[currently_training].step()
    collector.shutdown()
    del collector

    trained_agent.save()


def train_stage(
    combined_policy: CombinedPolicy,
    env: EnvBase,
    training_configs: tuple[TrainingConfig, TrainingConfig],
    device: torch.device | str,
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
            player,
            training_config,
            device,
            num_envs,
            pbar,
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
