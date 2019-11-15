import numpy as np
from pendulum import PendulumEnv


class ActorCritic(object):

    # TODO fill in this function to set up the Actor Critic model.
    # You may add extra parameters to this function to help with discretization
    # Also, you will need to tune the sigma value and learning rates
    def __init__(self, env, gamma=0.99, sigma=0.1, alpha_value=0.1, alpha_policy=0.1):
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

        # Step sizes for the value function and policy
        self.alpha_value = alpha_value
        self.alpha_policy = alpha_policy
        # These need to be tuned separately

    # TODO: fill in this function. 
    # This function should return an action given the
    # state by evaluating the Gaussian polic
    def act(self, state):
        pass

    # TODO: fill in this function that:
    #   1) Computes the value function gradient
    #   2) Computes the policy gradient
    #   3) Performs the gradient step for the value and policy functions
    # Given the (state, action, reward, next_state) from one step in the environment
    # You may return anything from this function to help with plotting/debugging
    def update(self, state, action, reward, next_state, done):
        pass


def train(env, model):
    num_episodes = 10000

    # TODO: write training and plotting code here
    for i in range(num_episodes):
        pass


if __name__ == "__main__":
    env = PendulumEnv()
    policy = ActorCritic(env)
    train(env, policy)

