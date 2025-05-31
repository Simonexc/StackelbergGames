import argparse
from collections import defaultdict
import multiprocessing
import yaml
import os
import time

import torch
from tensordict.nn.probabilistic import (
    InteractionType as ExplorationType,
    set_interaction_type as set_exploration_type,
)
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import Bounded
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import ActorValueOperator
from torchrl.objectives import LossModule
from torchrl.objectives.value import GAE
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from dotenv import dotenv_values

from environments.flipit_geometric import FlipItMap, FlipItEnv
from algorithms.simple_nn import Backbone, ActorHead, ValueHead, create_actor_value_operator, create_loss, CombinedPolicy


def create_defender(
    num_nodes: int, device: torch.device, gamma: float, lmbda: float, clip_epsilon: float, entropy_eps: float, lr: float
) -> tuple[ActorValueOperator, GAE, LossModule, torch.optim.Optimizer]:
    action_spec = Bounded(
        shape=torch.Size((1,)),
        low=0,
        high=num_nodes - 1,
        dtype=torch.int32,
    )

    defender_backbone = Backbone(
        num_nodes=num_nodes,
        embedding_size=32,
        player_type=0,
        device=device,
    )

    defender_actor_head = ActorHead(
        num_nodes=num_nodes,
        embedding_size=32,
        device=device,
    )

    defender_value_head = ValueHead(
        embedding_size=32,
        device=device,
    )

    defender_actor_value_operator = create_actor_value_operator(
        defender_backbone, defender_actor_head, defender_value_head, action_spec
    )

    defender_advantage = GAE(
        gamma=gamma, lmbda=lmbda, value_network=defender_actor_value_operator.get_value_operator()#, average_gae=True
    )

    defender_loss = create_loss(defender_actor_value_operator, clip_epsilon, entropy_eps)
    defender_optim = torch.optim.Adam(
        defender_loss.parameters(), lr=lr
    )

    return defender_actor_value_operator, defender_advantage, defender_loss, defender_optim


def create_attacker(
        num_nodes: int, device: torch.device, gamma: float, lmbda: float, clip_epsilon: float, entropy_eps: float, lr: float
) -> tuple[ActorValueOperator, GAE, LossModule, torch.optim.Optimizer]:
    action_spec = Bounded(
        shape=torch.Size((1,)),
        low=0,
        high=num_nodes - 1,
        dtype=torch.int32,
    )

    attacker_backbone = Backbone(
        num_nodes=num_nodes,
        embedding_size=32,
        player_type=1,
        device=device,
    )

    attacker_actor_head = ActorHead(
        num_nodes=num_nodes,
        embedding_size=32,
        device=device,
    )

    attacker_value_head = ValueHead(
        embedding_size=32,
        device=device,
    )

    attacker_actor_value_operator = create_actor_value_operator(
        attacker_backbone, attacker_actor_head, attacker_value_head, action_spec
    )

    attacker_advantage = GAE(
        gamma=gamma, lmbda=lmbda, value_network=attacker_actor_value_operator.get_value_operator()#, average_gae=True
    )

    attacker_loss = create_loss(attacker_actor_value_operator, clip_epsilon, entropy_eps)
    attacker_optim = torch.optim.Adam(
        attacker_loss.parameters(), lr=lr
    )

    return attacker_actor_value_operator, attacker_advantage, attacker_loss, attacker_optim


def training_loop(run: wandb.sdk.wandb_run.Run, device: torch.device):
    num_steps = run.config["environment"]["num_steps"]
    path_to_map = run.config["environment"]["path_to_map"]

    # Training specific settings
    lr = run.config["training"]["lr"]
    max_grad_norm = run.config["training"]["max_grad_norm"]
    steps_per_batch = run.config["training"]["steps_per_batch"]
    total_steps = run.config["training"]["total_steps"]
    sub_batch_size = run.config["training"]["steps_per_sub_batch"]  # size of sub-samples gathered from the data in the inner loop
    num_epochs = run.config["training"]["num_epochs"]  # number of epochs per batch of collected data

    # optimizer and advantage calculation
    clip_epsilon = run.config["training"]["clip_epsilon"]
    gamma = run.config["training"]["gamma"]
    lmbda = run.config["training"]["lmbda"]
    entropy_eps = run.config["training"]["entropy_eps"]

    flipit_map = FlipItMap.load(path_to_map)
    env = FlipItEnv(flipit_map, num_steps, device)

    defender_actor_value_operator, defender_advantage, defender_loss, defender_optim = create_defender(
        num_nodes=env.map.num_nodes,
        device=device,
        gamma=gamma,
        lmbda=lmbda,
        clip_epsilon=clip_epsilon,
        entropy_eps=entropy_eps,
        lr=lr,
    )
    defender_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        defender_optim, total_steps // (steps_per_batch * 2), 0.0
    )

    attacker_actor_value_operator, attacker_advantage, attacker_loss, attacker_optim = create_attacker(
        num_nodes=env.map.num_nodes,
        device=device,
        gamma=gamma,
        lmbda=lmbda,
        clip_epsilon=clip_epsilon,
        entropy_eps=entropy_eps,
        lr=lr,
    )
    attacker_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        attacker_optim, total_steps // (steps_per_batch * 2), 0.0
    )

    combined_policy = CombinedPolicy(
        defender_actor_value_operator.get_policy_operator(),
        attacker_actor_value_operator.get_policy_operator(),
    )

    collector = MultiaSyncDataCollector(
        [lambda: env for _ in range(cpu_cores)],
        combined_policy,
        frames_per_batch=steps_per_batch,
        total_frames=total_steps,
        split_trajs=False,
        device=device,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=steps_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantages = [defender_advantage, attacker_advantage]
    losses = [defender_loss, attacker_loss]
    losses[0].eval()
    losses[1].eval()
    optimizers = [defender_optim, attacker_optim]
    schedulers = [defender_scheduler, attacker_scheduler]
    names = ["defender", "attacker"]

    pbar = tqdm(total=total_steps)
    eval_str = ""
    defender_num_steps = 0
    attacker_num_steps = 0
    defender_num_epochs = 0
    attacker_num_epochs = 0

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        currently_training = i % 2
        losses[currently_training].train()
        current_player = names[currently_training]

        tensordict_data.update({
            "action": tensordict_data["action"][..., currently_training],
        })
        tensordict_data["next"].update({
            "reward": tensordict_data["next"]["reward"][..., currently_training].unsqueeze(-1),  # retain dimensionality
        })


        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantages[currently_training](tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            loss_objective_avg = []
            loss_critic_avg = []
            loss_entropy_avg = []
            state_value = []
            entropy = []
            advantage = []
            for current_step in range(steps_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = losses[currently_training](subdata.to(device))
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )
                currently_added_steps = steps_per_batch
                if current_step + steps_per_batch > sub_batch_size:
                    currently_added_steps = sub_batch_size - current_step

                if currently_training == 0:
                    defender_num_steps += currently_added_steps
                else:
                    attacker_num_steps += currently_added_steps

                run.log({
                    f"loss/objective_step_{current_player}": loss_vals["loss_objective"].item(),
                    f"loss/critic_step_{current_player}": loss_vals["loss_critic"].item(),
                    f"loss/entropy_step_{current_player}": loss_vals["loss_entropy"].item(),
                    f"train/entropy_step_{current_player}": loss_vals["entropy"].item(),
                    f"train/state_value_step_{current_player}": subdata["state_value"].mean().item(),
                    f"train/advantage_step_{current_player}": subdata["advantage"].mean().item(),
                    f"general/step_{current_player}": defender_num_steps if currently_training == 0 else attacker_num_steps,
                    f"general/step_total": defender_num_steps + attacker_num_steps,
                    f"general/epoch_{current_player}": defender_num_epochs if currently_training == 0 else attacker_num_epochs,
                    f"general/epoch_total": defender_num_epochs + attacker_num_epochs,
                    f"general/cycle_{current_player}": i // 2,
                    f"general/cycle_total": i,
                })
                loss_objective_avg.append(loss_vals["loss_objective"].item())
                loss_critic_avg.append(loss_vals["loss_critic"].item())
                loss_entropy_avg.append(loss_vals["loss_entropy"].item())
                state_value.append(subdata["state_value"].mean().item())
                entropy.append(loss_vals["entropy"].item())
                advantage.append(subdata["advantage"].mean().item())

                # Optimization: backward, grad clipping and optimization step
                optimizers[currently_training].zero_grad()
                loss_value.backward()
                # grad_norm = losses[currently_training].actor_network[-1].module[0].module.actor_head[-1].weight.grad.norm().item()
                # print(f"Gradient norm (actor final layer weight): {grad_norm:.6f}")
                # break
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(losses[currently_training].parameters(), max_grad_norm)
                optimizers[currently_training].step()

            if currently_training == 0:
                defender_num_epochs += 1
            else:
                attacker_num_epochs += 1

            run.log({
                f"loss/objective_{current_player}": sum(loss_objective_avg) / len(loss_objective_avg),
                f"loss/critic_{current_player}": sum(loss_critic_avg) / len(loss_critic_avg),
                f"loss/entropy_{current_player}": sum(loss_entropy_avg) / len(loss_entropy_avg),
                f"train/state_value_{current_player}": sum(state_value) / len(state_value),
                f"train/entropy_{current_player}": sum(entropy) / len(entropy),
                f"train/advantage_{current_player}": sum(advantage) / len(advantage),
                f"general/epoch_{current_player}": defender_num_epochs if currently_training == 0 else attacker_num_epochs,
                f"general/epoch_total": defender_num_epochs + attacker_num_epochs,
                f"general/cycle_{current_player}": i // 2,
                f"general/cycle_total": i,
            })

        reward_name = f"reward_{names[currently_training]}"
        lr_name = f"lr_{names[currently_training]}"
        rewards = tensordict_data["next", "reward"].mean().item()
        rewards_std = tensordict_data["next", "reward"].std().item()
        current_lr = optimizers[currently_training].param_groups[0]["lr"]

        run.log({
            f"train/{reward_name}_mean": rewards,
            f"train/{reward_name}_std": rewards_std,
            f"train/{lr_name}": current_lr,
            f"general/cycle_{names[currently_training]}": i // 2,
            f"general/cycle_total": i,
        })

        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward {names[currently_training]}={rewards: 4.4f}"
        )
        lr_str = f"lr {names[currently_training]} policy: {current_lr: 4.4f}"
        if i % 2 == 1:
            defender_actor_value_operator.eval()
            attacker_actor_value_operator.eval()
            with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
                eval_rollout = env.rollout(100, combined_policy)

                run.log({
                    "eval/reward_defender_mean": eval_rollout["next", "reward"][..., 0].mean().item(),
                    "eval/reward_defender_std": eval_rollout["next", "reward"][..., 0].std().item(),
                    "eval/reward_defender_sum": eval_rollout["next", "reward"][..., 0].sum().item(),
                    "eval/reward_attacker_mean": eval_rollout["next", "reward"][..., 1].mean().item(),
                    "eval/reward_attacker_std": eval_rollout["next", "reward"][..., 1].std().item(),
                    "eval/reward_attacker_sum": eval_rollout["next", "reward"][..., 1].sum().item(),
                    "eval/cycle_total": i,
                })
                del eval_rollout

        # losses[currently_training].eval()
        pbar.set_description(", ".join([cum_reward_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        schedulers[currently_training].step()
    collector.shutdown()
    time.sleep(2)
    del collector
    time.sleep(2)

    torch.save(defender_actor_value_operator.state_dict(), "defender_actor_value_operator.pth")


if __name__ == "__main__":
    # get arguements
    parser = argparse.ArgumentParser(description="Training script for FlipIt")
    parser.add_argument(
        "name",
        type=str,
        help="Name of run",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to config file",
    )
    args = parser.parse_args()

    env_config = dotenv_values("../.env")

    # General settings defining environment
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    cpu_cores = multiprocessing.cpu_count()
    print(f"Creating {cpu_cores} processes.")
    print(args.config, args.name)

    with open(args.config, "r") as file:
        config_content = yaml.safe_load(file)

    with wandb.init(
        entity=env_config["WANDB_ENTITY"],
        project=env_config["WANDB_PROJECT"],
        config=config_content,
        name=args.name,
    ) as run:
        training_loop(run, device)
