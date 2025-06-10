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
from config import TrainingConfig, LossConfig, EnvConfig, Player, AgentNNConfig, BackboneConfig, HeadConfig
from utils import train_agent


env_config = dotenv_values("../.env")

EMBEDDING_SIZE = 32

"""
https://www.ijcai.org/proceedings/2024/0880.pdf
https://arxiv.org/pdf/2306.01324
"""
def training_loop(device: torch.device, cpu_cores: int, player: int, run_name: str | None = None, config=None):
    player_name = Player(player).name
    if not run_name:
        run_name = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}-{player_name}'

    if config is None:
        wandb.init(
            entity=env_config["WANDB_ENTITY"],
            project=env_config["WANDB_PROJECT"],
            name=run_name,
        )
        config = wandb.config

    env_config_ = EnvConfig.from_dict(config)
    training_config = TrainingConfig.from_dict(config, suffix=f"_{player_name}")
    loss_config = LossConfig.from_dict(config, suffix=f"_{player_name}")
    agent_config = AgentNNConfig.from_dict(config)
    backbone_config = BackboneConfig.from_dict(config, suffix=f"_backbone")
    head_config = HeadConfig.from_dict(config, suffix=f"_head")

    flipit_map = FlipItMap.load(env_config_.path_to_map, device)
    env = FlipItEnv(flipit_map, env_config_.num_steps, device)

    num_nodes = flipit_map.num_nodes

    if player == 0:
        defender_agent = TrainableNNAgentPolicy(
            num_nodes=num_nodes,
            total_steps=env.num_steps,
            player_type=0,
            device=device,
            loss_config=loss_config,
            training_config=training_config,
            run_name=run_name,
            agent_config=agent_config,
            backbone_config=backbone_config,
            head_config=head_config,
        )
        attacker_agent = MultiAgentPolicy(
            num_nodes=num_nodes,
            player_type=1,
            device=device,
            policy_generator=AgentGenerator(
                NNAgentPolicy,
                {}
            ),
            run_name=run_name,
        )
    else:
        defender_agent = NNAgentPolicy(
            num_nodes=num_nodes,
            total_steps=env.num_steps,
            player_type=0,
            backbone_config=backbone_config,
            head_config=head_config,
            agent_config=agent_config,
            device=device,
            run_name=run_name,
        )
        attacker_agent = TrainableNNAgentPolicy(
            num_nodes=num_nodes,
            total_steps=env.num_steps,
            player_type=1,
            device=device,
            loss_config=loss_config,
            training_config=training_config,
            run_name=run_name,
            backbone_config=backbone_config,
            head_config=head_config,
            agent_config=agent_config,
            #scheduler_steps=training_config.total_steps_per_turn // training_config.steps_per_batch + 5,
        )

    combined_policy = CombinedPolicy(
        defender_agent,
        attacker_agent,
    )

    pbar = tqdm(total=training_config.total_steps_per_turn * training_config.player_turns)
    train_agent(
        combined_policy,
        env,
        player=player,
        training_config=training_config,
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
        "player",
        type=int,
        choices=[0, 1],
        help="Player to train: 0 for defender, 1 for attacker",
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

    run_name_ = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}-attacker-{args.name}'

    with wandb.init(
        entity=env_config["WANDB_ENTITY"],
        project=env_config["WANDB_PROJECT"],
        config=config_content,
        name=run_name_,
    ) as run:
        training_loop(device, cpu_cores, args.player, run_name=run_name_, config=run.config)
