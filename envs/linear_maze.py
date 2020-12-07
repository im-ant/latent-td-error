# ============================================================================
# Linear maze Markov Reward Process
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class LinearMazeMRP(gym.Env):
    """
    Description: deterministically move down a corridor

    State space:

    Action:
    """

    def __init__(self, feature_mat=None, seed=0):

        # Default settings
        self.n_states = 10
        self.feature_dim = self.n_states
        self.FeatureMat = np.zeros((self.n_states + 1, self.n_states))
        for i in range(self.n_states):
            self.FeatureMat[i, i] = 1.0

        # Initialize feature matrix if available
        if feature_mat is not None:
            self.FeatureMat = feature_mat
            self.n_states = np.shape(self.FeatureMat)[0] - 1
            self.feature_dim = np.shape(self.FeatureMat)[1]

        # ==
        # Initialize spaces
        self.action_space = gym.spaces.Discrete(n=1)  # dummy var
        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.feature_dim),
            high=np.ones(self.feature_dim),
            dtype=np.float
        )

        self.rng = np.random.default_rng(seed)
        self.state = 0

    def step(self, action):
        """
        Transition in the random walk chain
        :return:
        """
        # ==
        # Update state
        reward = 0.0
        done = False

        # Transition
        if self.state < self.n_states:
            self.state += 1

        # Feature, reward and termination
        phi = self.FeatureMat[self.state, :]
        if self.state >= self.n_states:
            reward = 1.0
            done = True

        return phi, reward, done, {}

    def reset(self):
        self.state = 0
        phi = self.FeatureMat[self.state, :]
        return phi

    def get_transition_matrix(self):
        """
        Helper function to return the transition matrix for this env
        :return: (self.n_states+1, self.n_states+1) np matrix
        """
        pass

    def get_reward_function(self):
        """
        Helper function to get the (tabular) reward function for each state
        NOTE: charactering each state's reward by the expected reward
              from transitioning out of the state. This may be slightly
              different from the TD way of solving for state values.
        :return: (self.n_states+1,) vector
        """
        pass

    def solve_linear_reward_parameters(self):
        """
        Helper function to solve for the best-fit linear parameters for the
        reward function.
        :return: (self.feature_dim, ) parameters for reward fn
        """
        pass

    def render(self):
        pass

    def close(self):
        pass


if __name__ == '__main__':

    seed = np.random.randint(100)
    print('numpy seed:', seed)
    # np.random.seed(seed)

    # TODO add back stuff about env and running env?
    env = LinearMazeMRP(seed=seed)

    # ==
    # Run a few
    print('=== set-up ===')
    print('env', env)
    print('action space', env.action_space)
    print('obs space', env.observation_space)

    print('=== start ===')
    cur_obs = env.reset()
    print(f's: {env.state}, obs: {cur_obs}')

    for step in range(30):
        action = env.action_space.sample()
        cur_obs, reward, done, info = env.step(action)

        print(f'a: {action}, s: {env.state}, obs: {cur_obs}, r:{reward}, d:{done}, {info}')
