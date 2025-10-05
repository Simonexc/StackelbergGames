import argparse
import multiprocessing
import yaml
from functools import partial
from datetime import datetime

import torch
from tqdm import tqdm
import wandb
from dotenv import dotenv_values

from algorithms.simple_nn import TrainableNNAgentPolicy, NNAgentPolicy
from algorithms.generic_policy import CombinedPolicy, MultiAgentPolicy, ExplorerAgent
from algorithms.keys_processors import CombinedExtractor
from algorithms.generator import AgentGenerator
from config import TrainingConfig, LossConfig, EnvConfig, Player, AgentNNConfig, BackboneConfig, HeadConfig
from utils import train_agent
from environments.env_mapper import EnvMapper


env_config = dotenv_values("../.env")

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
    try:
        agent_config = AgentNNConfig.from_dict(config)
        backbone_config = BackboneConfig.from_dict(config, suffix=f"_backbone")
        head_config = HeadConfig.from_dict(config, suffix=f"_head")
        if agent_config.embedding_size % backbone_config.num_head != 0 and backbone_config.cls_name == "BackboneTransformer":
            agent_config.embedding_size += backbone_config.num_head - (agent_config.embedding_size % backbone_config.num_head)

        agent_config_attacker = agent_config
        agent_config_defender = agent_config
        backbone_config_attacker = backbone_config
        backbone_config_defender = backbone_config
        head_config_attacker = head_config
        head_config_defender = head_config
    except TypeError:  # HACK: means that we have separate configs for attacker and defender
        agent_config_attacker = AgentNNConfig.from_dict(config, suffix="_attacker")
        agent_config_defender = AgentNNConfig.from_dict(config, suffix="_defender")
        backbone_config_attacker = BackboneConfig.from_dict(config, suffix="_backbone_attacker")
        backbone_config_defender = BackboneConfig.from_dict(config, suffix="_backbone_defender")
        head_config_attacker = HeadConfig.from_dict(config, suffix="_head_attacker")
        head_config_defender = HeadConfig.from_dict(config, suffix="_head_defender")
        if agent_config_attacker.embedding_size % backbone_config_attacker.num_head != 0 and backbone_config_attacker.cls_name == "BackboneTransformer":
            agent_config_attacker.embedding_size += backbone_config_attacker.num_head - (agent_config_attacker.embedding_size % backbone_config_attacker.num_head)
        if agent_config_defender.embedding_size % backbone_config_defender.num_head != 0 and backbone_config_defender.cls_name == "BackboneTransformer":
            agent_config_defender.embedding_size += backbone_config_defender.num_head - (agent_config_defender.embedding_size % backbone_config_defender.num_head)

    env_map, env = env_config_.create("cpu")

    defender_extractor = CombinedExtractor(player_type=0, env=env, actions_map=backbone_config_defender.extractors)
    attacker_extractor = CombinedExtractor(player_type=1, env=env, actions_map=backbone_config_attacker.extractors)

    if player == 0:
        defender_agent = TrainableNNAgentPolicy(
            player_type=0,
            max_sequence_size=env_config_.num_steps + 1,
            extractor=defender_extractor,
            action_size=env.action_size,
            env_type=EnvMapper.from_name(env_config_.env_name),
            device=device,
            loss_config=loss_config,
            training_config=training_config,
            run_name=run_name,
            agent_config=agent_config_defender,
            backbone_config=backbone_config_defender,
            head_config=head_config_defender,
            num_defenders=env.num_defenders,
            num_attackers=env.num_attackers,
        )
        attacker_agent = MultiAgentPolicy(
            action_size=env.action_size,
            player_type=1,
            device=device,
            embedding_size=agent_config_attacker.embedding_size,
            policy_generator=AgentGenerator(
                NNAgentPolicy,
                {}
            ),
            run_name=run_name,
            num_defenders=env.num_defenders,
            num_attackers=env.num_attackers,
        )
    else:
        defender_agent = NNAgentPolicy(
            player_type=0,
            max_sequence_size=env_config_.num_steps + 1,
            extractor=defender_extractor,
            action_size=env.action_size,
            backbone_config=backbone_config_defender,
            head_config=head_config_defender,
            agent_config=agent_config_defender,
            device=device,
            run_name=run_name,
            num_defenders=env.num_defenders,
            num_attackers=env.num_attackers,
        )
        attacker_agent = TrainableNNAgentPolicy(
            player_type=1,
            max_sequence_size=env_config_.num_steps + 1,
            extractor=attacker_extractor,
            action_size=env.action_size,
            env_type=EnvMapper.from_name(env_config_.env_name),
            device=device,
            loss_config=loss_config,
            training_config=training_config,
            run_name=run_name,
            backbone_config=backbone_config_attacker,
            head_config=head_config_attacker,
            agent_config=agent_config_attacker,
            num_defenders=env.num_defenders,
            num_attackers=env.num_attackers,
            # scheduler_steps=training_config.total_steps_per_turn // training_config.steps_per_batch + 5,
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
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, uses single GPU or CPU",
    )
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # General settings defining environment
    is_fork = multiprocessing.get_start_method() == "fork"

    # Parse GPU IDs if provided
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

    run_name_ = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}-attacker-{args.name}'

    with wandb.init(
        entity=env_config["WANDB_ENTITY"],
        project=env_config["WANDB_PROJECT"],
        config=config_content,
        name=run_name_,
    ) as run:
        training_loop(device, cpu_cores, args.player, run_name=run_name_, config=run.config)
