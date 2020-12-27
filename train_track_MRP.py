# =============================================================================
# Training agent on the linear track Markov Reward Process
#
# Author: Anthony G. Chen
# =============================================================================

import json
import numpy as np
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from envs.mrp import MarkovRewardProcess
from envs.linearTrackGenerator import LinearTrackEnvGenerator
from algos.td_lambda import TDLambdaAgent


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # ==
    # Set up the MRP parameter generator
    genCls = globals()[cfg.environment.generator.cls]
    envGen = genCls(**cfg.environment.generator.kwargs)

    # ==
    # Set up the MRP gym environment

    # Initialize feature matrix
    feature_mat = envGen.get_feature_matrix(
        feature_type=cfg.environment.feature.type,
        feature_args=cfg.environment.feature.args,
    )

    # Environment arguments
    env_kwargs = {
        'transition_mat': envGen.get_transition_matrix(),
        'feature_mat': feature_mat,
        'init_distribution': envGen.get_initial_distribution(),
        'reward_vec': envGen.get_reward_function(),
        'seed': cfg.environment.env_seed,
    }

    environment = MarkovRewardProcess(**env_kwargs)

    # ==
    # Set up agent
    agentCls = globals()[cfg.agent.cls]
    agent = agentCls(
        feature_dim=environment.observation_space.shape[0],
        **cfg.agent.kwargs,
    )

    # ==
    # Run training
    training_output_dict = run_single_training(agent, environment, cfg)

    # ==
    # Save output
    save_output(env_kwargs, training_output_dict, cfg)


def run_single_training(agent, env, cfg: DictConfig):
    # ==================================================
    # Logging
    training_dict = {
        'rewards': [],
        'td_errors': [],
        'parameters': [],
    }

    # ==================================================
    # Run experiment
    for episode_idx in range(cfg.training.num_episodes):
        # Reset env
        obs = env.reset()
        agent.begin_episode(obs)

        # Logging
        epis_obs = []
        epis_rewards = []
        steps = 0

        while True:
            # Interact with environment
            obs, reward, done, info = env.step(0)
            agent.step(obs, reward, done)

            # Tracker variables
            epis_rewards.append(reward)
            steps += 1

            if done:
                ag_td_errors = agent.log_dict['td_errors']
                print(episode_idx,
                      f'Cumu Reward: {np.sum(epis_rewards)}',
                      f'Cumu TD errors: {np.sum(ag_td_errors)}')

                # Save
                training_dict['rewards'].append(epis_rewards)
                training_dict['td_errors'].append(ag_td_errors)
                training_dict['parameters'].append(agent.get_parameters())

                # ==
                # Terminate
                break

    return training_dict


def save_output(env_kwargs, training_dict, cfg: DictConfig):
    # ==================================================
    # Save dictionary

    out_dict = {
        'config_dict': OmegaConf.to_container(cfg),
        'env_kwargs': env_kwargs,
        'training_data': training_dict,
    }

    save_data = True
    out_file_name = 'data.json'

    # ==
    # Json dump

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # elif hasattr(obj, 'tolist'):
        #    return obj.tolist()
        raise TypeError('Not serializable')

    if save_data:
        with open(out_file_name, 'w') as fp:
            json.dump(out_dict, fp, default=default)

        print(f'Output path:{os.getcwd()}/{out_file_name}')


if __name__ == "__main__":
    main()
