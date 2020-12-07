# =============================================================================
# TD(lambda) algorithm with eligibility traces from Sutton & Barto.
# Assume prediction only and no control.
#
# Author: Anthony G. Chen
# =============================================================================

import numpy as np

from algos.base import BaseLinearAgent


class TDLambdaAgent(BaseLinearAgent):
    def __init__(self, feature_dim,
                 gamma=0.9,
                 lamb=0.8,
                 lr=0.1,
                 seed=0):

        """
        TODO define arguments
        """
        super().__init__(feature_dim, gamma=gamma, lr=lr, seed=seed)
        self.lamb = lamb

        # Initialize V function and trace
        self.Wv = np.zeros(self.feature_dim)
        self.Z = np.zeros(self.feature_dim)

    def begin_episode(self, phi):
        super().begin_episode(phi)
        self.log_dict = {
            'td_errors': [],
        }

    def step(self, phi_t: np.array, reward: float, done: bool) -> None:
        """
        Take step in the environment
        :param phi_t: vector of feature observation
        :param reward: scalar reward
        :param done: boolean for whether episode is finished
        :return: integer action index
        """

        # Save trajectory
        if not done:
            self.traj['phi'].append(phi_t)
        self.traj['r'].append(reward)

        # ==
        # Learning (via trace)
        if len(self.traj['r']) > 0:
            self._optimize_model(done)

    def _optimize_model(self, done) -> None:
        # ==
        # Unpack current experience tuple, (S, A, R')
        t_idx = len(self.traj['r']) - 1
        cur_phi = self.traj['phi'][t_idx]
        rew = self.traj['r'][t_idx]

        # ==
        # Update trace
        self.Z = (self.lamb * self.gamma) * self.Z + cur_phi

        # ==
        # Update Q function
        if not done:
            nex_phi = self.traj['phi'][t_idx + 1]
            nex_v = np.dot(nex_phi, self.Wv)
        else:
            nex_v = 0.0

        # TD error
        cur_v = np.dot(cur_phi, self.Wv)
        td_err = rew + (self.gamma * nex_v) - cur_v

        # Parameter updates
        delta_Wv = td_err * self.Z
        self.Wv = self.Wv + (self.lr * delta_Wv)

        # ==
        # Logging losses
        self.log_dict['td_errors'].append(td_err)

    def report(self, logger, episode_idx):
        # Compute average predictions and log
        pass


# ==
# For testing purposes only
if __name__ == "__main__":
    agent = TDLambdaAgent(feature_dim=5)

    print(agent)
    print(agent.Wv)
