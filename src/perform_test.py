import argparse
import yaml
import os

from torchrl.envs.utils import check_env_specs
from dotenv import dotenv_values
import multiprocessing
import torch

from config import TrainingConfig, LossConfig, AgentNNConfig, BackboneConfig, HeadConfig, EnvConfig
from algorithms.coevosg import CoevoSGDefenderAgent, CoevoSGAttackerAgent, CoevoSGConfig
from algorithms.simple_nn import TrainableNNAgentPolicy
from algorithms.generic_policy import MultiAgentPolicy
from algorithms.generator import AgentGenerator
from algorithms.keys_processors import CombinedExtractor
from environments.env_mapper import EnvMapper
from algorithms.generic_policy import RandomAgent, GreedyOracleAgent, PoachersLogicModule, FlipItLogicModule, PoliceLogicModule
from utils import compare_agent_pairs
import wandb
import tempfile


if __name__ == "__main__":
    dot_env_config = dotenv_values("../.env")
    cpu_cores = min(6, multiprocessing.cpu_count())
    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nn_id",
        type=str,
    )
    parser.add_argument(
        "--gnn_id",
        type=str,
    )
    parser.add_argument(
        "--gnn_transformer_id",
        type=str,
    )
    parser.add_argument(
        "--coevosg_run",
        type=str,
    )
    args = parser.parse_args()

    coevosg_model_path = os.path.join(dot_env_config.get("MODELS_PATH", "."), "saved_models", args.coevosg_run)
    if not os.path.exists(coevosg_model_path):
        coevosg_model_path = os.path.join(".", "saved_models", args.coevosg_run)

    # Get configs from wandb runs and save as YAML files in temp directory
    temp_dir = tempfile.mkdtemp()
    wandb_api = wandb.Api()

    # Fetch and save NN model config
    nn_run = wandb_api.run(f"{dot_env_config.get('WANDB_ENTITY')}/{dot_env_config.get('WANDB_PROJECT')}/{args.nn_id}")
    nn_model_path = os.path.join(dot_env_config.get("MODELS_PATH", "."), "saved_models", nn_run.name)
    print(nn_model_path)
    nn_model_config_path = os.path.join(temp_dir, f"{args.nn_id}_config.yaml")
    with open(nn_model_config_path, "w") as f:
        yaml.dump(dict(nn_run.config), f)

    # Fetch and save GNN model config
    gnn_run = wandb_api.run(f"{dot_env_config.get('WANDB_ENTITY')}/{dot_env_config.get('WANDB_PROJECT')}/{args.gnn_id}")
    gnn_model_path = os.path.join(dot_env_config.get("MODELS_PATH", "."), "saved_models", gnn_run.name)
    print(gnn_model_path)
    gnn_model_config_path = os.path.join(temp_dir, f"{args.gnn_id}_config.yaml")
    with open(gnn_model_config_path, "w") as f:
        yaml.dump(dict(gnn_run.config), f)

    # Fetch and save GNN Transformer model config
    gnn_transformer_run = wandb_api.run(f"{dot_env_config.get('WANDB_ENTITY')}/{dot_env_config.get('WANDB_PROJECT')}/{args.gnn_transformer_id}")
    gnn_transformer_model_path = os.path.join(dot_env_config.get("MODELS_PATH", "."), "saved_models", gnn_transformer_run.name)
    print(gnn_transformer_model_path)
    gnn_transformer_model_config_path = os.path.join(temp_dir, f"{args.gnn_transformer_id}_config.yaml")
    with open(gnn_transformer_model_config_path, "w") as f:
        yaml.dump(dict(gnn_transformer_run.config), f)

    # Load NN
    with open(nn_model_config_path, "r") as file:
        nn_config = yaml.safe_load(file)
    env_config = EnvConfig.from_dict(nn_config)

    env_map, env = env_config.create("cpu")
    check_env_specs(env)

    nn_training_config_defender = TrainingConfig.from_dict(nn_config, suffix="_defender")
    nn_loss_config_defender = LossConfig.from_dict(nn_config, suffix="_defender")
    nn_training_config_attacker = TrainingConfig.from_dict(nn_config, suffix="_attacker")
    nn_loss_config_attacker = LossConfig.from_dict(nn_config, suffix="_attacker")
    try:
        nn_agent_config = AgentNNConfig.from_dict(nn_config)
        nn_backbone_config = BackboneConfig.from_dict(nn_config, suffix=f"_backbone")
        nn_head_config = HeadConfig.from_dict(nn_config, suffix=f"_head")

        nn_agent_config_attacker = nn_agent_config
        nn_agent_config_defender = nn_agent_config
        nn_backbone_config_attacker = nn_backbone_config
        nn_backbone_config_defender = nn_backbone_config
        nn_head_config_attacker = nn_head_config
        nn_head_config_defender = nn_head_config
    except TypeError:  # HACK: means that we have separate configs for attacker and defender
        nn_agent_config_attacker = AgentNNConfig.from_dict(nn_config, suffix="_attacker")
        nn_agent_config_defender = AgentNNConfig.from_dict(nn_config, suffix="_defender")
        nn_backbone_config_attacker = BackboneConfig.from_dict(nn_config, suffix="_backbone_attacker")
        nn_backbone_config_defender = BackboneConfig.from_dict(nn_config, suffix="_backbone_defender")
        nn_head_config_attacker = HeadConfig.from_dict(nn_config, suffix="_head_attacker")
        nn_head_config_defender = HeadConfig.from_dict(nn_config, suffix="_head_defender")

    nn_defender_extractor = CombinedExtractor(player_type=0, env=env, actions_map=nn_backbone_config_defender.extractors)
    nn_defender_agent = TrainableNNAgentPolicy(
        player_type=0,
        max_sequence_size=env_config.num_steps + 1,
        extractor=nn_defender_extractor,
        action_size=env.action_size,
        env_type=EnvMapper.from_name(env_config.env_name),
        device=device,
        loss_config=nn_loss_config_defender,
        training_config=nn_training_config_defender,
        agent_config=nn_agent_config_defender,
        backbone_config=nn_backbone_config_defender,
        head_config=nn_head_config_defender,
        run_name="test",
        add_logs=False,  # No logs during testing
        num_attackers=env.num_attackers,
        num_defenders=env.num_defenders,
    )
    nn_defender_agent.eval()
    nn_defender_agent.load(os.path.join(nn_model_path, "defender", "agent_0.pth"))

    nn_attacker_extractor = CombinedExtractor(player_type=1, env=env, actions_map=nn_backbone_config_attacker.extractors)
    nn_attacker_agent = MultiAgentPolicy(
        action_size=env.action_size,
        player_type=1,
        device=device,
        run_name="test",
        embedding_size=nn_agent_config_attacker.embedding_size,
        policy_generator=AgentGenerator(
            TrainableNNAgentPolicy,
            {
                "extractor": nn_attacker_extractor,
                "max_sequence_size": env_config.num_steps + 1,
                "action_size": env.action_size,
                "env_type": EnvMapper.from_name(env_config.env_name),
                "player_type": 1,
                "device": device,
                "loss_config": nn_loss_config_attacker,
                "training_config": nn_training_config_attacker,
                "run_name": "test",
                "add_logs": False,  # No logs during testing
                "agent_config": nn_agent_config_attacker,
                "backbone_config": nn_backbone_config_attacker,
                "head_config": nn_head_config_attacker,
                "num_attackers": env.num_attackers,
                "num_defenders": env.num_defenders,
            }
        ),
        num_attackers=env.num_attackers,
        num_defenders=env.num_defenders,
    )
    nn_attacker_agent.eval()
    nn_attacker_agent.load(os.path.join(nn_model_path, "attacker"))

    # Load GNN
    with open(gnn_model_config_path, "r") as file:
        gnn_config = yaml.safe_load(file)

    gnn_training_config_defender = TrainingConfig.from_dict(gnn_config, suffix="_defender")
    gnn_loss_config_defender = LossConfig.from_dict(gnn_config, suffix="_defender")
    gnn_training_config_attacker = TrainingConfig.from_dict(gnn_config, suffix="_attacker")
    gnn_loss_config_attacker = LossConfig.from_dict(gnn_config, suffix="_attacker")
    try:
        gnn_agent_config = AgentNNConfig.from_dict(gnn_config)
        gnn_backbone_config = BackboneConfig.from_dict(gnn_config, suffix=f"_backbone")
        gnn_head_config = HeadConfig.from_dict(gnn_config, suffix=f"_head")

        gnn_agent_config_attacker = gnn_agent_config
        gnn_agent_config_defender = gnn_agent_config
        gnn_backbone_config_attacker = gnn_backbone_config
        gnn_backbone_config_defender = gnn_backbone_config
        gnn_head_config_attacker = gnn_head_config
        gnn_head_config_defender = gnn_head_config
    except TypeError:  # HACK: means that we have separate configs for attacker and defender
        gnn_agent_config_attacker = AgentNNConfig.from_dict(gnn_config, suffix="_attacker")
        gnn_agent_config_defender = AgentNNConfig.from_dict(gnn_config, suffix="_defender")
        gnn_backbone_config_attacker = BackboneConfig.from_dict(gnn_config, suffix="_backbone_attacker")
        gnn_backbone_config_defender = BackboneConfig.from_dict(gnn_config, suffix="_backbone_defender")
        gnn_head_config_attacker = HeadConfig.from_dict(gnn_config, suffix="_head_attacker")
        gnn_head_config_defender = HeadConfig.from_dict(gnn_config, suffix="_head_defender")

    gnn_defender_extractor = CombinedExtractor(player_type=0, env=env, actions_map=gnn_backbone_config_defender.extractors)
    gnn_defender_agent = TrainableNNAgentPolicy(
        player_type=0,
        max_sequence_size=env_config.num_steps + 1,
        extractor=gnn_defender_extractor,
        action_size=env.action_size,
        env_type=EnvMapper.from_name(env_config.env_name),
        device=device,
        loss_config=gnn_loss_config_defender,
        training_config=gnn_training_config_defender,
        agent_config=gnn_agent_config_defender,
        backbone_config=gnn_backbone_config_defender,
        head_config=gnn_head_config_defender,
        run_name="test",
        add_logs=False,  # No logs during testing
        num_attackers=env.num_attackers,
        num_defenders=env.num_defenders,
    )
    gnn_defender_agent.eval()
    gnn_defender_agent.load(os.path.join(gnn_model_path, "defender", "agent_0.pth"))

    gnn_attacker_extractor = CombinedExtractor(player_type=1, env=env, actions_map=gnn_backbone_config_attacker.extractors)
    gnn_attacker_agent = MultiAgentPolicy(
        action_size=env.action_size,
        player_type=1,
        device=device,
        run_name="test",
        embedding_size=gnn_agent_config_attacker.embedding_size,
        policy_generator=AgentGenerator(
            TrainableNNAgentPolicy,
            {
                "extractor": gnn_attacker_extractor,
                "max_sequence_size": env_config.num_steps + 1,
                "action_size": env.action_size,
                "env_type": EnvMapper.from_name(env_config.env_name),
                "player_type": 1,
                "device": device,
                "loss_config": gnn_loss_config_attacker,
                "training_config": gnn_training_config_attacker,
                "run_name": "test",
                "add_logs": False,  # No logs during testing
                "agent_config": gnn_agent_config_attacker,
                "backbone_config": gnn_backbone_config_attacker,
                "head_config": gnn_head_config_attacker,
                "num_attackers": env.num_attackers,
                "num_defenders": env.num_defenders,
            }
        ),
        num_attackers=env.num_attackers,
        num_defenders=env.num_defenders,
    )
    gnn_attacker_agent.eval()
    gnn_attacker_agent.load(os.path.join(gnn_model_path, "attacker"))

    # Load GNN Transformer
    with open(gnn_transformer_model_config_path, "r") as file:
        gnn_transformer_config = yaml.safe_load(file)

    gnn_transformer_training_config_defender = TrainingConfig.from_dict(gnn_transformer_config, suffix="_defender")
    gnn_transformer_loss_config_defender = LossConfig.from_dict(gnn_transformer_config, suffix="_defender")
    gnn_transformer_training_config_attacker = TrainingConfig.from_dict(gnn_transformer_config, suffix="_attacker")
    gnn_transformer_loss_config_attacker = LossConfig.from_dict(gnn_transformer_config, suffix="_attacker")
    try:
        gnn_transformer_agent_config = AgentNNConfig.from_dict(gnn_config)
        gnn_transformer_backbone_config = BackboneConfig.from_dict(gnn_config, suffix=f"_backbone")
        gnn_transformer_head_config = HeadConfig.from_dict(gnn_config, suffix=f"_head")

        gnn_transformer_agent_config_attacker = gnn_transformer_agent_config
        gnn_transformer_agent_config_defender = gnn_transformer_agent_config
        gnn_transformer_backbone_config_attacker = gnn_transformer_backbone_config
        gnn_transformer_backbone_config_defender = gnn_transformer_backbone_config
        gnn_transformer_head_config_attacker = gnn_transformer_head_config
        gnn_transformer_head_config_defender = gnn_transformer_head_config
    except TypeError:  # HACK: means that we have separate configs for attacker and defender
        gnn_transformer_agent_config_attacker = AgentNNConfig.from_dict(gnn_transformer_config, suffix="_attacker")
        gnn_transformer_agent_config_defender = AgentNNConfig.from_dict(gnn_transformer_config, suffix="_defender")
        gnn_transformer_backbone_config_attacker = BackboneConfig.from_dict(gnn_transformer_config, suffix="_backbone_attacker")
        gnn_transformer_backbone_config_defender = BackboneConfig.from_dict(gnn_transformer_config, suffix="_backbone_defender")
        gnn_transformer_head_config_attacker = HeadConfig.from_dict(gnn_transformer_config, suffix="_head_attacker")
        gnn_transformer_head_config_defender = HeadConfig.from_dict(gnn_transformer_config, suffix="_head_defender")

    gnn_transformer_defender_extractor = CombinedExtractor(player_type=0, env=env, actions_map=gnn_transformer_backbone_config_defender.extractors)
    gnn_transformer_defender_agent = TrainableNNAgentPolicy(
        player_type=0,
        max_sequence_size=env_config.num_steps + 1,
        extractor=gnn_transformer_defender_extractor,
        action_size=env.action_size,
        env_type=EnvMapper.from_name(env_config.env_name),
        device=device,
        loss_config=gnn_transformer_loss_config_defender,
        training_config=gnn_transformer_training_config_defender,
        agent_config=gnn_transformer_agent_config_defender,
        backbone_config=gnn_transformer_backbone_config_defender,
        head_config=gnn_transformer_head_config_defender,
        run_name="test",
        add_logs=False,  # No logs during testing
        num_attackers=env.num_attackers,
        num_defenders=env.num_defenders,
    )
    gnn_transformer_defender_agent.eval()
    gnn_transformer_defender_agent.load(os.path.join(gnn_transformer_model_path, "defender", "agent_0.pth"))

    gnn_transformer_attacker_extractor = CombinedExtractor(player_type=1, env=env, actions_map=gnn_transformer_backbone_config_attacker.extractors)
    gnn_transformer_attacker_agent = MultiAgentPolicy(
        action_size=env.action_size,
        player_type=1,
        device=device,
        run_name="test",
        embedding_size=gnn_transformer_agent_config_attacker.embedding_size,
        policy_generator=AgentGenerator(
            TrainableNNAgentPolicy,
            {
                "extractor": gnn_transformer_attacker_extractor,
                "max_sequence_size": env_config.num_steps + 1,
                "action_size": env.action_size,
                "env_type": EnvMapper.from_name(env_config.env_name),
                "player_type": 1,
                "device": device,
                "loss_config": gnn_transformer_loss_config_attacker,
                "training_config": gnn_transformer_training_config_attacker,
                "run_name": "test",
                "add_logs": False,  # No logs during testing
                "agent_config": gnn_transformer_agent_config_attacker,
                "backbone_config": gnn_transformer_backbone_config_attacker,
                "head_config": gnn_transformer_head_config_attacker,
                "num_attackers": env.num_attackers,
                "num_defenders": env.num_defenders,
            }
        ),
        num_attackers=env.num_attackers,
        num_defenders=env.num_defenders,
    )
    gnn_transformer_attacker_agent.eval()
    gnn_transformer_attacker_agent.load(os.path.join(gnn_transformer_model_path, "attacker"))

    # Load CoevoSG
    coevosg_defender_agent = CoevoSGDefenderAgent(
        device="cpu",
        run_name="test",
        config=CoevoSGConfig(),
        env=env,
    )
    coevosg_defender_agent.eval()
    coevosg_defender_agent.load(os.path.join(coevosg_model_path, "defender", "agent_0.pth"))

    coevosg_attacker_agent = CoevoSGAttackerAgent(
        device="cpu",
        run_name="test",
        config=CoevoSGConfig(),
        env=env,
    )
    coevosg_attacker_agent.eval()
    coevosg_attacker_agent.load(os.path.join(coevosg_model_path, "attacker", "agent_0.pth"))

    # Opponents
    if env_config.env_name == "flipit":
        logic_module = FlipItLogicModule(env=env, player_type=1, total_steps=env.num_steps, device="cpu")
    elif env_config.env_name == "poachers":
        logic_module = PoachersLogicModule(env=env, player_type=1, total_steps=env.num_steps, device="cpu")
    elif env_config.env_name == "police":
        logic_module = PoliceLogicModule(env=env, player_type=1, total_steps=env.num_steps, device="cpu")
    else:
        raise ValueError(f"Unknown environment name: {env_config.env_name}")

    random_attacker_agent = RandomAgent(
        action_size=env.action_size,
        embedding_size=nn_agent_config_attacker.embedding_size,
        player_type=1,
        device="cpu",
        run_name="test",
        num_attackers=env.num_attackers,
        num_defenders=env.num_defenders,
    )
    greedy_oracle_attacker_agent = GreedyOracleAgent(
        action_size=env.action_size,
        embedding_size=nn_agent_config_attacker.embedding_size,
        total_steps=env.num_steps,
        player_type=1,
        device="cpu",
        run_name="test",
        map_logic=logic_module,
        num_attackers=env.num_attackers,
        num_defenders=env.num_defenders,
    )
    print("processing")
    results = compare_agent_pairs(
        [
            (nn_defender_agent, nn_attacker_agent, "nn"),
            (gnn_defender_agent, gnn_attacker_agent, "gnn"),
            (gnn_transformer_defender_agent, gnn_transformer_attacker_agent, "gnn_transformer"),
            (coevosg_defender_agent, coevosg_attacker_agent, "coevosg"),
        ],
        [
            (random_attacker_agent, "random"),
            (greedy_oracle_attacker_agent, "greedy"),
        ],
        env,
        print_results=False,
    )

    output_file = os.path.join(dot_env_config.get("RESULTS_PATH", "."), "stackelberg", f"test_{env_config.env_name}_{env_config.num_nodes}_{env_config.num_steps}_{env_config.seed}.yaml")
    print(output_file)
    with open(output_file, "w") as f:
        yaml.dump(results, f)
