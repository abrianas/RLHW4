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
def tabular_epsilon_greedy_policy(Q, eps, state, rand_flag): ###
    # if rand_flag=True then no random actions are selected
    n_actions = Q.shape[1]
    if rand_flag or np.random.rand() > eps:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, n_actions)
    return action


class QLearning(object):
    # Initialize a Qlearning object
    # alpha is the "learning_rate"
    def __init__(self, num_states, num_actions, alpha=0.5, gamma=0.9):
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
    
    for e in range(niter):
        state = env.reset()
        total = 0 # stores the sum of rewards per episode
        t = 0
        num_goals = 0 # stores the number of times the goal is reached in niter-steps
        done = False
        while True:
            t += 1
            action = tabular_epsilon_greedy_policy(qlearning.Q, eps, state, True) # we want NO randomness
            next_state, reward, done = env.step(action)
            total += reward
            qlearning.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                tstep_rewards.append(total)
                break
                
    for i in range(len(tstep_rewards)):
        if tstep_rewards[i] > 0:
            num_goals += 1
            
    # returns the fraction of times that the policy reached the goal
    frac = num_goals/len(tstep_rewards)
    return tstep_rewards, frac


# this function will take in an environment(GridWorld),
# a Qlearning object,
# an epsilon in the range [0, 1],
# and the number of episodes you want to run the algorithm for
# Returns the sum of the rewards at each timestep for each episode
def offpolicyTD(env, qlearning, num_episodes, eps):
    tstep_rewards = []
    qvalues = []
    
    
    
    for e in range(num_episodes):
        state = env.reset()
        episode_log = []
        total = 0 # stores the sum of rewards at each timestep per episode
        done = False
        while True:
            action = tabular_epsilon_greedy_policy(qlearning.Q, eps, state, False) # we want randomness
            next_state, reward, done = env.step(action)
            
            # append results to the episode log
            episode_log.append([state, action])
            
            total += reward
            qlearning.update(state, action, reward, next_state, done)
            state = next_state
            
            # if done, an episode has been complete, store the results for later            
            if done:
                # collects the Q-value at the start state for each episode
                episode_log = np.array(episode_log)
                for i in range(len(episode_log)):
                    if episode_log[i,0]==0: # if at the starting state
                        qvalues.append(qlearning.Q[episode_log[i,0],episode_log[i,1]])
                        break
                
                # collects the sum of the rewards at each timestep per episode
                tstep_rewards.append(total)
                
                # finds the optimal Q-value 
                if total>0: ### NEEDS TO BE FIXED
                    q_optimal = qlearning.Q[state,action]
                    #print(q_optimal)
                break

    return tstep_rewards, np.asarray(qvalues), q_optimal

if __name__ == "__main__":
    num_episodes = 1000
    eps = 0.1
    env = GridWorld(MAP2)
    # for question 1
    # qlearning = QLearning(env.get_num_states(), env.get_num_actions())
    # [tstep_rewards,_,_] = offpolicyTD(env, qlearning, num_episodes, eps)
    # plt.plot(tstep_rewards)
    # plt.xlabel("Number of Episodes")
    # plt.ylabel("Total Rewards")
    # plt.title("eps-Greedy w/ Randomness")
    # plt.show()

    # # evaluate the greedy policy to see how well it performs
    # [_,frac] = evaluate_greedy_policy(qlearning, env)
    # print("Finding goal " + str(frac*100) + "% of the time.")

    # for question 2
    qlearning = QLearning(env.get_num_states(), env.get_num_actions())
    [_, qvalue, q_optimal] = offpolicyTD(env, qlearning, num_episodes, eps)
    plt.plot(qvalue)
    plt.axhline(y=q_optimal, color='r', linestyle='-') ### how do we know what the optimal Q value is?
    plt.xlabel("Number of Episodes")
    plt.ylabel("Q Values")
    plt.title("eps-Greedy w/ Randomness")
    plt.show()


