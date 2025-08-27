import argparse
import multiprocessing
import yaml
from datetime import datetime

import torch
from tqdm import tqdm
import wandb
from dotenv import dotenv_values

from algorithms.simple_nn import TrainableNNAgentPolicy
from algorithms.generic_policy import CombinedPolicy, MultiAgentPolicy
from algorithms.generator import AgentGenerator
from algorithms.keys_processors import CombinedExtractor
from config import TrainingConfig, LossConfig, EnvConfig, AgentNNConfig, BackboneConfig, HeadConfig
from utils import train_stage
from environments.env_mapper import EnvMapper


env_config = dotenv_values("../.env")

"""
https://www.ijcai.org/proceedings/2024/0880.pdf
https://arxiv.org/pdf/2306.01324
"""
def training_loop(device: torch.device, cpu_cores: int, run_name: str | None = None, config=None, log_wandb: bool = False):
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
    agent_config = AgentNNConfig.from_dict(config)
    backbone_config = BackboneConfig.from_dict(config, suffix=f"_backbone")
    head_config = HeadConfig.from_dict(config, suffix=f"_head")

    assert training_config_attacker.player_turns == training_config_defender.player_turns

    env_map, env = env_config_.create("cpu")

    defender_extractor = CombinedExtractor(player_type=0, env=env, actions_map=backbone_config.extractors)
    defender_agent = TrainableNNAgentPolicy(
        player_type=0,
        max_sequence_size=env_config_.num_steps + 1,
        extractor=defender_extractor,
        action_size=env.action_size,
        env_type=EnvMapper.from_name(env_config_.env_name),
        device=device,
        loss_config=loss_config_defender,
        training_config=training_config_defender,
        agent_config=agent_config,
        backbone_config=backbone_config,
        head_config=head_config,
        run_name=run_name,
        add_logs=log_wandb,  # Defender logs during training
    )
    attacker_extractor = CombinedExtractor(player_type=1, env=env, actions_map=backbone_config.extractors)
    attacker_agent = MultiAgentPolicy(
        action_size=env.action_size,
        player_type=1,
        device=device,
        run_name=run_name,
        embedding_size=agent_config.embedding_size,
        policy_generator=AgentGenerator(
            TrainableNNAgentPolicy,
            {
                "extractor": attacker_extractor,
                "max_sequence_size": env_config_.num_steps + 1,
                "action_size": env.action_size,
                "env_type": EnvMapper.from_name(env_config_.env_name),
                "player_type": 1,
                "device": device,
                "loss_config": loss_config_attacker,
                "training_config": training_config_attacker,
                "run_name": run_name,
                "add_logs": False,  # Attacker does not log during training
                "agent_config": agent_config,
                "backbone_config": backbone_config,
                "head_config": head_config,
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
            num_envs=cpu_cores,
            pbar=pbar,
        )

    return defender_agent, attacker_agent


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
    cpu_cores = min(12, multiprocessing.cpu_count())
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
        training_loop(device, cpu_cores, run_name_, run.config, log_wandb=True)
