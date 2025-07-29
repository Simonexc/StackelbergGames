import argparse
import multiprocessing
from functools import partial

from train_single_stage import training_loop

import torch
import wandb
from dotenv import dotenv_values


if __name__ == "__main__":
    # get arguements
    parser = argparse.ArgumentParser(description="Training script for FlipIt")
    parser.add_argument(
        "sweep",
        type=str,
        help="Path to config file",
    )

    args = parser.parse_args()
    
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    env_config = dotenv_values("../.env")

    # General settings defining environment
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    cpu_cores = min(multiprocessing.cpu_count(), 12)
    print(f"Creating {cpu_cores} processes.")

    train = partial(
        training_loop,
        device=device,
        cpu_cores=cpu_cores,
        player=1,  # Attacker
        run_name=None,  # Will be set by wandb
    )

    wandb.agent(args.sweep, train, count=None, entity=env_config["WANDB_ENTITY"], project=env_config["WANDB_PROJECT"])
