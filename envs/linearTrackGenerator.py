# ============================================================================
# Generator class for parameters of linear track MRP environments.
#
# Author: Anthony G. Chen
# ============================================================================

import gym
import numpy as np


class LinearTrackEnvGenerator:
    """
    Description: generate the parameters of a linear track environment
    """

    def __init__(self, n_states, seed=0):
        self.n_states = n_states  # default 10?
        self.rng = np.random.default_rng(seed)

    def get_initial_distribution(self):
        """
        Generate the initial distribution. For now always start on the left-
        most state.
        :return: (self.n_states, ) np matrix
        """
        initDistVec = np.zeros(self.n_states)
        initDistVec[0] = 1.0
        return initDistVec

    def get_transition_matrix(self):
        """
        Generate the transition matrix. For now always deterministically
        transition down the corridor.

        :return: (self.n_states, self.n_states) np matrix
        """
        transMat = np.zeros((self.n_states, self.n_states))
        # Deterministic transitions
        for i in range((self.n_states - 1)):
            transMat[i, i + 1] = 1.0
        transMat[-1, -1] = 1.0  # absorbing state at the end

        return transMat

    def get_reward_function(self):
        """
        Generate the reward function. For now always sparse reward at the
        last transition.
        :return: (self.n_states, ) np vector
        """
        rewardVec = np.zeros(self.n_states)
        rewardVec[-2] = 1.0
        return rewardVec

    def get_feature_matrix(self, feature_type: str,
                           feature_args: dict):
        """
        Generate the feature matrix
        :return: (self.n_states, d) np matrix
        """
        if feature_type == 'tabular':
            return self.gen_tabular_features(feature_args)
        elif feature_type == 'random':
            return self.gen_random_features(feature_args)
        elif feature_type == 'grid':
            return self.gen_grid_features(feature_args)
        elif feature_type == 'tile':
            return self.gen_tile_features(feature_args)
        else:
            raise ValueError

    def gen_tabular_features(self, args: dict):
        """
        Generate the tabular feature matrix
        :return: (n_states, n_states) np matrix
        """
        phi_mat = np.identity(self.n_states)
        phi_mat[-1] *= 0.0
        return phi_mat

    def gen_random_features(self, args: dict):
        """
        Generate random features
        :return: (n_states, d) np matrix
        """
        if args is None or len(args) == 0:
            args = {
                'feature_dim': 4,
            }

        phi_mat = self.rng.normal(
            loc=0.0, scale=1.0,
            size=(self.n_states, args['feature_dim'])
        )
        phi_mat[-1] *= 0.0
        return phi_mat

    def gen_grid_features(self, args: dict):
        """
        Generate grid-cell like features
        :return: (n_states, d) np matrix
        """
        if args is None or len(args) == 0:
            args = {
                'num_cells': 3,  # how many grid cells
                'first_cell_loc': 1,  # (ordered) first cell location idx
                'cell_spacing': 2,  # spacing between cells
                'grid_period': 5,  # spacing between grid field means
                'field_amplitude': 1.0,  # amplitude of grid field
                'field_scale': 0.9,  # sigma of Gaussian grid field
            }

        # ==
        # Initialize matrix
        phi_mat = np.empty((self.n_states, args['num_cells']))

        # ==
        # Generate grid cell fields

        cur_cell_loc = args['first_cell_loc']
        fAmp = args['field_amplitude']
        fSig = args['field_scale']

        for cell_idx in range(args['num_cells']):
            # Check within bound
            if cur_cell_loc >= self.n_states:
                raise IndexError

            # Generate all the fields for current cell
            cumuFieldVec = np.zeros(self.n_states)
            cur_field_mu = cur_cell_loc  # tracks current field mean
            while cur_field_mu < self.n_states:
                xVec = np.arange(self.n_states)
                curFieldVec = (
                        fAmp * np.exp(-((xVec - cur_field_mu) ** 2 /
                                        (2 * (fSig ** 2))))
                )  # Gaussian activation given current field mean

                # Aggregate and increment to next field
                cumuFieldVec = np.maximum(cumuFieldVec, curFieldVec)
                cur_field_mu += args['grid_period']

            # Store fields of current cell, increment to next cell
            phi_mat[:, cell_idx] = cumuFieldVec
            cur_cell_loc += args['cell_spacing']

        phi_mat[-1] *= 0.0
        return phi_mat

    def gen_tile_features(self, args: dict):
        """
        Generate tile-tiling features, as seen in
        Sutton & Barto, chapter 9.5.4
        :return: (n_states, d) np matrix
        """
        raise NotImplementedError


if __name__ == '__main__':
    seed = np.random.randint(100)
    print('numpy seed:', seed)
    # np.random.seed(seed)

    env_generator = LinearTrackEnvGenerator(n_states=10, seed=seed)
    feature_args = None

    np.set_printoptions(precision=3)

    print('Init distribution')
    p = env_generator.get_initial_distribution()
    print(p)

    print('Transition matrix')
    p = env_generator.get_transition_matrix()
    print(p)

    print('Reward function')
    p = env_generator.get_reward_function()
    print(p)

    print('Tabular')
    hmat = env_generator.get_feature_matrix('tabular', feature_args)
    print(hmat)

    print('Random')
    hmat = env_generator.get_feature_matrix('random', feature_args)
    print(hmat)

    print('Grid')
    hmat = env_generator.get_feature_matrix('grid', feature_args)
    print(hmat)

    print('Tile')
    hmat = env_generator.get_feature_matrix('tile', feature_args)
    print(hmat)
