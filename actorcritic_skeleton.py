import numpy as np
from pendulum_v2 import PendulumEnv

import pdb
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler



class ActorCritic(object):

    # TODO fill in this function to set up the Actor Critic model.
    # You may add extra parameters to this function to help with discretization
    # Also, you will need to tune the sigma value and learning rates
    def __init__(self, env,num_states, num_actions, no_rbf, gamma=0.99, sigma=0.9, alpha_value=0.01, alpha_policy=0.001):
        # Upper and lower limits of the state

        self.min_state = env.min_state
        self.max_state = env.max_state

        # TODO initialize the table for the value function
        self.value = np.zeros((0, ))
        # TODO initialize the table for the mean value of the policy
        self.mean_policy = np.zeros((0, ))

        # Discount factor (don't tune)
        self.gamma = gamma

        # Standard deviation of the policy (need to tune)
        self.sigma = sigma
        self.no_rbf = no_rbf

        # Step sizes for the value function and policy
        self.alpha_value = alpha_value
        self.alpha_policy = alpha_policy
        # These need to be tuned separately
        self.num_states = num_states
        self.num_actions = num_actions
        # here are the weights for the policy - you may change this initialization
        self.theta = np.zeros((self.no_rbf, self.num_actions))
        self.weights = np.zeros(self.no_rbf)




    # TODO: fill in this function.
    # This function should return an action given the
    # state by evaluating the Gaussian polic
    def act(self, state):
        action = np.random.normal(np.dot(state,self.theta),self.sigma)

        return action

    # TODO: fill in this function that:
    #   1) Computes the value function gradient
    #   2) Computes the policy gradient
    #   3) Performs the gradient step for the value and policy functions
    # Given the (state, action, reward, next_state) from one step in the environment
    # You may return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done, I):
        if done:
            value_next_state = 0
        else:
            value_next_state = np.matmul(next_state,self.weights.T)

        value_state = np.matmul(state,self.weights.T)

        advantage = reward + self.gamma*value_next_state - value_state
        gradientV = state
        grad = ((action - np.dot(state,self.theta))/self.sigma**2)*state

        self.weights += self.alpha_value*advantage*np.reshape(gradientV,[-1,1])[0]

        self.theta +=   self.alpha_policy*I*advantage*np.reshape(grad,[-1,1])


def train(env,policy):

    num_episodes = 10000

    # TODO: write training and plotting code here
    for i in range(num_episodes):
        I = 1
        state = env.reset()

        state = state_normalize(state)
        feature_state = rbf(state, centers, rbf_sigma)
        # feature_state = featurizer.transform([state])
        done = False
        print(i)
        while done == False:

            action = policy.act(feature_state)
            env.render()
            next_state, reward, done, blah= env.step(action)

            # next_state = compute2dstate(next_state)

            next_state = state_normalize(next_state)
            feature_next_state = rbf(next_state, centers, rbf_sigma)
            # feature_next_state = featurizer.transform([next_state])
            policy.update(feature_state, action, reward, feature_next_state, done, I)
            I = policy.gamma*I
            feature_state = feature_next_state

    env.close()
    return


# def compute2dstate(state):
#
#     theta = np.arctan2(state[1],state[0])
#     state_2d = np.array([theta, state[2]])
#     return state_2d

def state_normalize(state):

    return np.array([state[0],state[1],state[2]/8.0])



def rbf(state, centers, rbf_sigma):
    ## input state, return rbf features pi
    phi = []
    for c in range(0, len(centers)):
        rbf_eval =  np.exp(-np.linalg.norm(state - centers[c,:])**2/(2*(rbf_sigma**2)))
        phi.append(rbf_eval)


    return np.asarray(phi)
#
def computeRBFcenters(high,low,no_rbf):
    theta_cos = np.linspace(low[0],high[0], no_rbf)
    theta_sin = np.linspace(low[1],high[1], no_rbf)
    theta_dot = np.linspace(low[2],high[2], no_rbf)
    theta_cos_c, theta_sin_c, theta_dot_c = np.meshgrid(theta_cos,theta_sin, theta_dot)
    centers = []

    for i in range(0,no_rbf):
        for j in range(0,no_rbf):
            for k in range(0,no_rbf):
                c = [theta_cos_c[i,j,k],theta_sin_c[i,j,k], theta_dot_c[i,j,k]]
                centers.append(c)

    centers = np.asarray(centers)

    return centers




if __name__ == "__main__":
    env = PendulumEnv()

    high = [1.0,1.0,1.0]
    low = [-1.0,-1.0,-1.0]
    no_rbf = 3
    rbf_sigma = 1.0/(no_rbf - 1)

    centers = computeRBFcenters(high, low, no_rbf)
    no_centers = len(centers)
    # env.reset()
    # observation_examples = []
    # for i in range(300):
    # 	s,r,d,_ = env.step([1.])
    #
    #     # s = compute2dstate(s)
    # 	observation_examples.append(s)
    #
    # # Create radial basis function sampler to convert states to features for nonlinear function approx
    # featurizer = sklearn.pipeline.FeatureUnion([
    #         ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    #         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    #         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    #         ("rbf4", RBFSampler(gamma=0.5, n_components=100))
    # 		])
    # # Fit featurizer to our samples
    # featurizer.fit(np.array(observation_examples))
    # no_centers = 400
    policy = ActorCritic(env,3,1,no_centers)
    train(env, policy)
