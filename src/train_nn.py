import argparse
import multiprocessing
import yaml
from datetime import datetime

import torch
from tqdm import tqdm
import wandb
from dotenv import dotenv_values

from environments.flipit_geometric import FlipItMap, FlipItEnv
from algorithms.simple_nn import TrainableNNAgentPolicy, NNAgentPolicy
from algorithms.generic_policy import CombinedPolicy, MultiAgentPolicy
from algorithms.generator import AgentGenerator
from config import TrainingConfig, LossConfig, EnvConfig
from utils import train_stage


env_config = dotenv_values("../.env")

"""
https://www.ijcai.org/proceedings/2024/0880.pdf
https://arxiv.org/pdf/2306.01324
"""
def training_loop(device: torch.device, cpu_cores: int, run_name: str | None = None, config=None):
    if not run_name:
        run_name = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}-full'
    if config is None:
        wandb.init(
            entity=env_config["WANDB_ENTITY"],
            project=env_config["WANDB_PROJECT"],
            name=run_name,
        )
        config = wandb.config

    env_config_ = EnvConfig.from_dict(config)
    training_config_defender = TrainingConfig.from_dict(config, suffix="_defender")
    loss_config_defender = LossConfig.from_dict(config, suffix="_defender")
    training_config_attacker = TrainingConfig.from_dict(config, suffix="_attacker")
    loss_config_attacker = LossConfig.from_dict(config, suffix="_attacker")

    assert training_config_attacker.player_turns == training_config_defender.player_turns

    flipit_map = FlipItMap.load(env_config_.path_to_map)
    env = FlipItEnv(flipit_map, env_config_.num_steps, device)

    num_nodes = flipit_map.num_nodes

    defender_agent = TrainableNNAgentPolicy(
        num_nodes=num_nodes,
        player_type=0,
        embedding_size=32,
        device=device,
        loss_config=loss_config_defender,
        training_config=training_config_defender,
        run_name=run_name,
    )
    attacker_agent = MultiAgentPolicy(
        action_size=num_nodes,
        player_type=1,
        device=device,
        embedding_size=32,
        run_name=run_name,
        policy_generator=AgentGenerator(
            TrainableNNAgentPolicy,
            {
                "num_nodes": num_nodes,
                "player_type": 1,
                "embedding_size": 32,
                "device": device,
                "loss_config": loss_config_attacker,
                "training_config": training_config_attacker,
                "run_name": run_name,
            }
        ),
    )

    combined_policy = CombinedPolicy(
        defender_agent,
        attacker_agent,
    )

    pbar = tqdm(total=(
        training_config_defender.total_steps_per_turn * training_config_defender.player_turns + training_config_attacker.total_steps_per_turn * training_config_attacker.player_turns
    ))

    for turn in range(training_config_defender.player_turns):
        train_stage(
            combined_policy,
            env,
            training_configs=(training_config_defender, training_config_attacker),
            device=device,
            num_envs=cpu_cores,
            pbar=pbar,
        )


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(description="Training script for FlipIt")
    parser.add_argument(
        "config",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Name of run",
    )
    args = parser.parse_args()

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

    with open(args.config, "r") as file:
        config_content = yaml.safe_load(file)

    run_name_ = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}-full-{args.name}'
    with wandb.init(
        entity=env_config["WANDB_ENTITY"],
        project=env_config["WANDB_PROJECT"],
        config=config_content,
        name=run_name_,
    ) as run:
        training_loop(device, cpu_cores, run_name_, run.config)
