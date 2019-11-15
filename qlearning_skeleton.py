import numpy as np
from grid_world import *
import matplotlib.pyplot as plt
import math
import gym

# TODO: Fill this function in
# Function that takes an a 2d numpy array Q (num_states by num_actions)
# an epsilon in the range [0, 1] and the state
# to output actions according to an Epsilon-Greedy policy
# (random actions are chosen with epsilon probability)
def tabular_epsilon_greedy_policy(Q, eps, state, train=False): ###
    # if train=true then no random actions selected
    n_actions = Q.shape[1]
    if train or np.random.rand() < eps:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, n_actions)
        
    return action


class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, alpha=0.5, gamma=0.9):
        # initialize Q values to something
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma


    # TODO: fill in this function
    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step
    # you can return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done):
        # a = np.argmax(self.Q[next_state, :]) # should I use np.argmax or np.max?
        if done: # at the terminal state Q[next_state,a]=0
            self.Q[state,action] += self.alpha*(reward-self.Q[state,action])
        else:
            self.Q[state,action] += self.alpha*(reward+(self.gamma*np.max(self.Q[next_state, 
                :]))-Q[state,action])
            
        return



# TODO: fill this in
# run the greedy policy (no randomness) on the environment for niter number of times
# and return the fraction of times that it reached the goal
def evaluate_greedy_policy(qlearning, env, niter=100):
    pass


if __name__ == "__main__":
    env = GridWorld(MAP3)
    qlearning = QLearning(env.get_num_states(), env.get_num_actions())
    # must create a function to train the code

    # ## TODO: write training code here
    # num_episodes = 100
    # eps = 0.1
    # for i in range(num_episodes):
        # pass


    # # evaluate the greedy policy to see how well it performs
    # frac = evaluate_greedy_policy(qlearning, env)
    # print("Finding goal " + str(frac) + "% of the time.")



