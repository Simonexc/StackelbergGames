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


env_config = dotenv_values("../.env")

"""
https://www.ijcai.org/proceedings/2024/0880.pdf
https://arxiv.org/pdf/2306.01324
"""


def model_factory(env, backbone_config, env_config, training_config, agent_config, head_config, loss_config, run_name, player, device):
    defender_extractor = CombinedExtractor(player_type=0, env=env, actions_map=backbone_config.extractors)
    attacker_extractor = CombinedExtractor(player_type=1, env=env, actions_map=backbone_config.extractors)

    if player == 0:
        defender_agent = TrainableNNAgentPolicy(
            player_type=0,
            max_sequence_size=env_config.num_steps + 1,
            extractor=defender_extractor,
            action_size=env.action_size,
            env_type=env_config.env_pair,
            device=device,
            loss_config=loss_config,
            training_config=training_config,
            run_name=run_name,
            agent_config=agent_config,
            backbone_config=backbone_config,
            head_config=head_config,
        )
        attacker_agent = MultiAgentPolicy(
            action_size=env.action_size,
            player_type=1,
            device=device,
            embedding_size=agent_config.embedding_size,
            policy_generator=AgentGenerator(
                NNAgentPolicy,
                {}
            ),
            run_name=run_name,
        )
    else:
        defender_agent = NNAgentPolicy(
            player_type=0,
            max_sequence_size=env_config.num_steps + 1,
            extractor=defender_extractor,
            action_size=env.action_size,
            backbone_config=backbone_config,
            head_config=head_config,
            agent_config=agent_config,
            device=device,
            run_name=run_name,
        )
        attacker_agent = TrainableNNAgentPolicy(
            player_type=1,
            max_sequence_size=env_config.num_steps + 1,
            extractor=attacker_extractor,
            action_size=env.action_size,
            env_type=env_config.env_pair,
            device=device,
            loss_config=loss_config,
            training_config=training_config,
            run_name=run_name,
            backbone_config=backbone_config,
            head_config=head_config,
            agent_config=agent_config,
            #scheduler_steps=training_config.total_steps_per_turn // training_config.steps_per_batch + 5,
        )

    return CombinedPolicy(
        defender_agent,
        attacker_agent,
        #exploration_defender=exploration_defender,
        #exploration_attacker=exploration_attacker,
    )

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
    if agent_config.embedding_size % backbone_config.num_head != 0 and backbone_config.cls_name == "BackboneTransformer":
        agent_config.embedding_size += backbone_config.num_head - (agent_config.embedding_size % backbone_config.num_head)

    env_map, env = env_config_.create("cpu")

    factory = partial(model_factory, env, backbone_config, env_config_, training_config, agent_config, head_config, loss_config, run_name, player, device)
    combined_policy = factory()

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
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # General settings defining environment
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    cpu_cores = min(8, multiprocessing.cpu_count())
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
