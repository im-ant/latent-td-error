# =============================================================================
# Training agent on Markov Reward Process
#
# Author: Anthony G. Chen
# =============================================================================

import json
import numpy as np

from algos.td_lambda import TDLambdaAgent
from envs.linear_maze import LinearMazeMRP


def train_single_agent():

    # ==================================================
    # Initialization
    envCls = LinearMazeMRP
    env = envCls()

    # Initialize agent
    agentCls = TDLambdaAgent
    agent_kwargs = {
        'gamma': 0.9,
        'lamb': 0.0,
        'lr': 0.1,
        'seed': 0,
    }
    agent = agentCls(
        feature_dim=env.observation_space.shape[0],
        **agent_kwargs,
    )

    # ==================================================
    # Logging
    training_dict = {
        'features': [],
        'rewards': [],
        'td_errors': [],
    }

    # ==================================================
    # Run experiment
    for episode_idx in range(50):
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
            epis_obs.append(list(obs))
            epis_rewards.append(reward)
            steps += 1

            if done:
                ag_td_errors = agent.log_dict['td_errors']
                print(episode_idx,
                      f'Cumu Reward: {np.sum(epis_rewards)}',
                      f'Cumu TD errors: {np.sum(ag_td_errors)}')

                # Save
                training_dict['features'].append(epis_obs)
                training_dict['rewards'].append(epis_rewards)
                training_dict['td_errors'].append(ag_td_errors)

                # ==
                # Terminate
                break

    # ==================================================
    # Save dictionary
    data_dict = {
        'envCls_name': envCls.__name__,
        'agentCls_name': agentCls.__name__,
        'agent_kwargs': agent_kwargs,
        'feature_dim': env.observation_space.shape[0],
        'training_data': training_dict,
    }

    save_data = True
    out_file_path = 'data.json'
    if save_data:
        with open(out_file_path, 'w') as fp:
            json.dump(data_dict, fp)


if __name__ == "__main__":
    print('Start')
    train_single_agent()