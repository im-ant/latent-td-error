# ============================================================================
# General class for Markov Reward Processes
# For episodic setting, assume absorbing states indicate temrination.
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class MarkovRewardProcess(gym.Env):
    """
    Description: general class for running a Markov Reward process with
                 finite states.

    State space:

    Action:
    """

    def __init__(self,
                 transition_mat,
                 feature_mat,
                 init_distribution,
                 reward_vec,
                 seed=0):

        # Set
        self.tran_Mat = transition_mat  # (N, N)
        self.feat_Mat = feature_mat  # (N, d)
        self.init_Dis = init_distribution  # (N, )
        self.rew_Vec = reward_vec  # (N, )

        self.rng = np.random.default_rng(seed)

        #
        self.n_states = len(self.tran_Mat)  # N
        self.feature_dim = np.shape(self.feat_Mat)[1]  # d

        # ==
        # Set up gym spaces
        self.action_space = gym.spaces.Discrete(n=1)  # dummy var
        self.observation_space = gym.spaces.Box(
            low=(np.ones(self.feature_dim) * np.min(self.feat_Mat)),
            high=(np.ones(self.feature_dim) * np.max(self.feat_Mat)),
            dtype=np.float
        )

        # ==
        # Set up environment

        # Sample initial state
        self.state = self.rng.choice(self.n_states, p=self.init_Dis)

    def step(self, action):
        """
        Sample next state
        :return:
        """

        # Compute reward of leaving current state
        reward = self.rew_Vec[self.state]

        # Transition
        nex_state = self.rng.choice(self.n_states,
                                    p=self.tran_Mat[self.state, :])
        self.state = nex_state

        # Compute termination via absorbing state
        done = False
        if self.tran_Mat[self.state, self.state] == 1.0:
            done = True

        # Feature
        phi = self.feat_Mat[self.state, :]

        return phi, reward, done, {}

    def reset(self):
        self.state = self.rng.choice(self.n_states, p=self.init_Dis)
        phi = self.feat_Mat[self.state, :]
        return phi

    def get_transition_matrix(self):
        """
        Helper function to return the transition matrix for this env
        :return: (self.n_states+1, self.n_states+1) np matrix
        """
        return self.tran_Mat

    def get_reward_function(self):
        """
        Helper function to get the (tabular) reward function for each state
        NOTE: charactering each state's reward by the expected reward
              from transitioning out of the state. This may be slightly
              different from the TD way of solving for state values.
        :return: (self.n_states+1,) vector
        """
        return self.rew_Vec

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

    # Sample
    rng = np.random.default_rng(seed)
    n_states = 4

    initD = np.zeros(n_states)
    initD[0] = 1.0

    tMat = np.zeros((n_states, n_states))
    for i in range((n_states-1)):
        tMat[i, i+1] = 1.0
    tMat[-1, -1] = 1.0

    rVec = np.zeros(n_states)
    rVec[-2] = 1.0

    fMat = np.identity(n_states)
    fMat[-1] *= 0.0


    #
    env = MarkovRewardProcess(transition_mat=tMat,
                              feature_mat=fMat,
                              init_distribution=initD,
                              reward_vec=rVec,
                              seed=seed)


    np.set_printoptions(precision=3)
    print(env.init_Dis)
    print(env.tran_Mat)
    print(env.rew_Vec)
    print(env.feat_Mat)

    # ==
    # Run

    # ==
    # Run a few
    print('=== set-up ===')
    print('env', env)
    print('action space', env.action_space)
    print('obs space', env.observation_space)

    print('=== start ===')
    cur_obs = env.reset()
    print(f's: {env.state}, obs: {cur_obs}')

    for step in range(6):
        action = env.action_space.sample()
        cur_obs, reward, done, info = env.step(action)

        print(f'a: {action}, s: {env.state}, obs: {cur_obs}, r:{reward}, d:{done}, {info}')
