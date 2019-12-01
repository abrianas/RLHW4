from pendulum_v2 import PendulumEnv
import numpy as np
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
import copy
import pdb
import gym

#Hyperparameters
NUM_EPISODES = 10000
LEARNING_RATE = 0.000025
GAMMA = 0.99

# Create gym and seed numpy
env = gym.make('CartPole-v0')
nA = env.action_space.n
np.random.seed(1)
env = PendulumEnv()

# Init weight
w = np.random.rand(400, 2)

# Keep stats for final print of graph
episode_rewards = []

# Begin gathering samples to fit SKLearn featurizer
env.reset()

observation_examples = []
for i in range(300):
	s,r,d,_ = env.step([1.])
	observation_examples.append(s)

# Create radial basis function sampler to convert states to features for nonlinear function approx
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
		])
# Fit featurizer to our samples
featurizer.fit(np.array(observation_examples))

# Our policy that maps state to action parameterized by w
def policy(state,w):
    z = state.dot(w)

    exp = np.exp(z)
    return exp/np.sum(exp)

# Call this method on every state to transform it into higher-dimensional space
def featurize_state(state):
	# Transform states

    featurized = featurizer.transform([state])
    return featurized

# Vectorized softmax Jacobian
def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

# Main loop
# Make sure you update your weights AFTER each episode
for e in range(NUM_EPISODES):

	state = env.reset()
	state = featurize_state(state)

	grads = []
	rewards = []

	# Keep track of game score to print
	score = 0

	while True:

		# Uncomment to see your model train in real time (slower)
		#env.render()

		# Sample from policy and take action in environment
		probs = policy(state,w)
		action = np.array([np.random.choice(nA,p=probs[0])])
		next_state,reward,done,_ = env.step(action)
		next_state = featurize_state(next_state)

		# Compute gradient and save with reward in memory for our weight updates
		dsoftmax = softmax_grad(probs)[action,:]
		dlog = dsoftmax / probs[0,action]
		grad = state.T.dot(dlog[None,:])

		grads.append(grad)
		rewards.append(reward)


		score+=reward

		# Dont forget to update your old state to the new state
		state = next_state

		if done:
			break

        # Weight update
    for i in range(len(grads)):
        # Loop through everything that happend in the episode and update towards the log policy gradient times **FUTURE** reward

        w += LEARNING_RATE * grads[i] * sum([ r * (GAMMA ** r) for t,r in enumerate(rewards[i:])])

	# Append for logging and print
    episode_rewards.append(score)
    print(score, e)

plt.plot(np.arange(num_episodes),episode_rewards)
plt.show()
env.close()
