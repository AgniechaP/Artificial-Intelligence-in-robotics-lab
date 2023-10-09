# prob.py
# This is

import random
import numpy as np

from gridutil import *


class LocAgent:

    def __init__(self, size, walls, gold, pits, eps_move):
        self.size = size
        self.walls = walls
        # list of valid locations
        self.locations = list({*locations(self.size)}.difference(self.walls))
        # dictionary from location to its index in the list
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        self.eps_move = eps_move
        # whether to plan next direction to move
        self.plan_next_move = True
        # planned direction
        self.next_dir = None

        # previous action
        self.prev_action = None

        self.t = 0

        self.V = np.zeros([len(self.locations)], dtype=np.float)
        self.pi = ['N' for _ in self.locations]

        self.comp_value_and_policy(gold, pits)

    def comp_value_and_policy(self, gold, pits):
        reward_gold = 10
        reward_pits = -100
        gamma = 0.9
        eps_V = 1e-6 #granica do sprawdzenia czy jestesmy wystarczajaco podobni if |prev_V - V| < epsilon

        iter = 0
        
        # compute self.V and self.pi
        # PUT YOUR CODE HERE
        # Pseudocode:
        # converged = False
        # while not converged:
        #     prev_V = V
        #     for s in self.locations:
        #         best_V = np.inf
        #         best_a = 'N'
        #
        #         for a in ['N', 'E', 'S', 'W']:
        #             for sprim in next_states:













        # ------------------

        print('Policy found after ', iter, ' iterations')

    def get_policy(self):
        pi_dict = {loc: self.pi[i] for i, loc in enumerate(self.locations)}
        return pi_dict

    def __call__(self, percept, loc):
        # choose action according to policy
        action = self.pi[self.loc_to_idx[loc]]

        return action

    def forward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    def backward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    @staticmethod
    def turnright(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 1) % 4
        return cur_loc, dirs[idx]

    @staticmethod
    def turnleft(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 4 - 1) % 4
        return cur_loc, dirs[idx]
