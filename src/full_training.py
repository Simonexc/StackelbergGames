import torch
import os
import yaml
import wandb
from datetime import datetime
import argparse
import multiprocessing
from dotenv import dotenv_values

from train_nn import training_loop as training_loop_nn
from train_coevosg import training_loop as training_loop_coevosg

from algorithms.generic_policy import RandomAgent, GreedyOracleAgent
from config import EnvConfig
from utils import compare_agent_pairs


env_config = dotenv_values("../.env")


def training_loop(device: torch.device, cpu_cores: int, run_name: str | None = None, config=None, log_wandb: bool = False):
    """
    Full training loop that trains both the defender and attacker agents.
    """

    with open(os.path.join("configs", "run", "test_single_training_flipit_transformer.yaml"), "r") as file:
        config_transformer = yaml.safe_load(file)
        config_transformer.update(config)

    with open(os.path.join("configs", "run", "test_single_training_flipit.yaml"), "r") as file:
        config_fnn = yaml.safe_load(file)
        config_fnn.update(config)

    # Train defender agent using neural network policy
    defender_trans, attacker_trans = training_loop_nn(device, cpu_cores, run_name+"transformer", config_transformer, log_wandb)
    defender_fnn, attacker_fnn = training_loop_nn(device, cpu_cores, run_name+"fnn", config_fnn)

    # Train attacker agent using CoevoSG policy
    #defender_coevosg, attacker_coevosg = training_loop_coevosg("cpu", cpu_cores, run_name+"coevosg", config_fnn)

    env_config = EnvConfig.from_dict(config)
    env_map, env = env_config.create(device)

    random_agent = RandomAgent(action_size=env.action_size, embedding_size=32, player_type=1, device="cpu", run_name="test")
    greedy_oracle_agent = GreedyOracleAgent(action_size=env.action_size, embedding_size=32, total_steps=env.num_steps, player_type=1, device="cpu", run_name="test", env_map=env_map)
    defender_trans = defender_trans.to("cpu")
    attacker_trans = attacker_trans.to("cpu")
    defender_fnn = defender_fnn.to("cpu")
    attacker_fnn = attacker_fnn.to("cpu")

    results = compare_agent_pairs(
        [
            (defender_trans, attacker_trans, "transformer"),
            (defender_fnn, attacker_fnn, "fnn"),
            #(defender_coevosg, attacker_coevosg, "coevosg"),
        ],
        [
            (random_agent, "random"),
            (greedy_oracle_agent, "greedy"),
        ],
        env,
        print_results=True,
    )
    wandb.log(results)


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
        training_loop(device, cpu_cores, run_name_, run.config, log_wandb=True)
