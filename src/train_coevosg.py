import argparse
import multiprocessing
import yaml
from datetime import datetime

import torch
from tqdm import tqdm
import wandb
from dotenv import dotenv_values

from environments.flipit_geometric import FlipItMap, FlipItEnv
from algorithms.simple_nn import CombinedPolicy
from algorithms.coevosg2 import CoevoSGAttackerAgent, CoevoSGDefenderAgent
from algorithms.generator import AgentGenerator
from config import EnvConfig, CoevoSGConfig
from utils import train_stage_coevosg


env_config = dotenv_values("../.env")

"""
https://www.ijcai.org/proceedings/2024/0880.pdf
https://arxiv.org/pdf/2306.01324
"""
def training_loop(device: torch.device, cpu_cores: int, run_name: str | None = None, config=None):
    if not run_name:
        run_name = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}-full-coevosg'
    if config is None:
        wandb.init(
            entity=env_config["WANDB_ENTITY"],
            project=env_config["WANDB_PROJECT"],
            name=run_name,
        )
        config = wandb.config

    env_config_ = EnvConfig.from_dict(config)
    coevosg_config = CoevoSGConfig()  # CoevoSGConfig.from_dict(config)

    flipit_map = FlipItMap.load(env_config_.path_to_map)
    env = FlipItEnv(flipit_map, env_config_.num_steps, device)

    num_nodes = flipit_map.num_nodes

    defender_agent = CoevoSGDefenderAgent(
        action_size=num_nodes,
        player_type=0,
        embedding_size=32,
        device=device,
        run_name=run_name,
        config=coevosg_config,
        env=env,
    )
    attacker_agent = CoevoSGAttackerAgent(
        action_size=num_nodes,
        player_type=1,
        embedding_size=32,
        device=device,
        run_name=run_name,
        config=coevosg_config,
        env=env,
    )

    combined_policy = CombinedPolicy(
        defender_agent,
        attacker_agent,
    )

    num_turns = coevosg_config.generations // coevosg_config.gen_per_switch
    pbar = tqdm(total=num_turns)
    defender_agent.evaluate_population(attacker_agent.population)
    attacker_agent.evaluate_population(defender_agent.population)
    best_fitness = defender_agent.best_population.fitness
    no_improvement = 0

    for turn in range(num_turns):
        train_stage_coevosg(combined_policy, pbar)
        new_best_fitness = defender_agent.best_population.fitness
        print(new_best_fitness, best_fitness)
        if new_best_fitness > best_fitness:
            best_fitness = new_best_fitness
            no_improvement = 0
        else:
            no_improvement += coevosg_config.gen_per_switch

        if no_improvement >= coevosg_config.no_improvement_limit:
            print(f"No improvement for {no_improvement} generations, switching roles.")
            break

    defender_agent.save()
    attacker_agent.save()


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

    run_name_ = f'{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}-full-coevosg-{args.name}'
    with wandb.init(
        entity=env_config["WANDB_ENTITY"],
        project=env_config["WANDB_PROJECT"],
        config=config_content,
        name=run_name_,
    ) as run:
        training_loop(device, cpu_cores, run_name_, run.config)
