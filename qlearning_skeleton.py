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
def tabular_epsilon_greedy_policy(Q, eps, state, train_flag): ###
    # if train_flag=True then no random actions are selected
    n_actions = Q.shape[1]
    if train_flag or np.random.rand() > eps: ## rename the train variable
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, n_actions)
    return action


class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, alpha=0.5, gamma=0.8):
        # initialize Q values to something
        #self.Q = np.zeros((num_states, num_actions))
        self.Q = np.random.random((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma


    # TODO: fill in this function
    # updates the Q value table
    # with a (state, action, reward, next_state) from one step in the environment
    # done is a bool indicating if the episode terminated at this step
    # you can return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done):
        a = np.argmax(self.Q[next_state, :])
        if done: # at the terminal state Q[next_state,a]=0
            self.Q[state,action] += self.alpha*(reward-self.Q[state,action])
        else:
            self.Q[state,action] += self.alpha*(reward+(self.gamma*self.Q[next_state,a])-self.Q[state,action])



# TODO: fill this in
# run the greedy policy (no randomness) on the environment for niter number of times
# and return the fraction of times that it reached the goal
def evaluate_greedy_policy(qlearning, env, niter=100):
    tstep_rewards = [] 
    
    for e in range(num_episodes):
        state = env.reset()
        total = 0 # stores the sum of rewards per episode
        t = 0
        goal = 0 # stores the number of times the goal is reached in niter-steps
        done = False
        while t<niter:
            t += 1
            action = tabular_epsilon_greedy_policy(qlearning.Q, eps, state, True) # we want NO randomness
            next_state, reward, done = env.step(action)
            total += reward
            if reward > 0: # if the goal is reached, the reward will be positive
                goal += 1
            qlearning.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                tstep_rewards.append(total)
                break
    
    # returns the fraction of times that the policy reached the goal
    frac = goal/niter
    return tstep_rewards, frac


# this function will take in an environment(GridWorld),
# a Qlearning object,
# an epsilon in the range [0, 1],
# and the number of episodes you want to run the algorithm for
# Returns the sum of the rewards at each timestep for each episode
def offpolicyTD(env, qlearning, num_episodes, eps):
    tstep_rewards = [] 
    
    for e in range(num_episodes):
        state = env.reset()
        total = 0 # stores the sum of rewards per timestep
        done = False
        while True:
            action = tabular_epsilon_greedy_policy(qlearning.Q, eps, state, False) # we want randomness
            next_state, reward, done = env.step(action)
            total += reward
            qlearning.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                tstep_rewards.append(total)
                break
        
    # must print out iteration number for one of the questions    
    return tstep_rewards

if __name__ == "__main__":
    num_episodes = 1000
    eps = 0.1
    env = GridWorld(MAP3)
    qlearning = QLearning(env.get_num_states(), env.get_num_actions())
    tstep_rewards = offpolicyTD(env, qlearning, num_episodes, eps)
    plt.plot(tstep_rewards)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Total Rewards")
    plt.title("eps-Greedy w/ Randomness")
    plt.show()

    # # evaluate the greedy policy to see how well it performs
    # frac = evaluate_greedy_policy(qlearning, env)
    # print("Finding goal " + str(frac) + "% of the time.")



