# =============================================================================
# Linear Agent Base Class, prediction only
#
# Author: Anthony G. Chen
# =============================================================================

from typing import List, Tuple

from gym import spaces
import numpy as np


class BaseLinearAgent(object):
    def __init__(self, feature_dim, gamma=0.9, lr=0.1, seed=0):
        """
        TODO define arguments
        """
        self.feature_dim = feature_dim  # feature dimension
        self.gamma = gamma
        self.lr = lr  # step size / learning rate

        # Saving single-episode trajectory
        self.traj = None

        # Log
        self.log_dict = None

        # RNG
        self.rng = np.random.default_rng(seed)

    def begin_episode(self, phi_0):
        """
        Start of episode
        :param observation: integer denoting tabular state index
        :return: integer action index
        """

        # Initialize trajectory
        self.traj = {
            'phi': [phi_0],
            'r': []
        }

        return None

    def step(self, phi_t: np.array, reward: float, done: bool) -> int:
        """
        Take step in the environment
        :param phi_t: vector of feature observation
        :param reward: scalar reward
        :param done: boolean for whether episode is finished
        :return: integer action index
        """
        pass

    def _optimize_model(self, done) -> None:
        # Optimize
        # Log losses
        pass

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = BaseAgent()
    print(agent)
