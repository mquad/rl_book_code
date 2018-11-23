"""
Code to replicate the examples of Figure 2.2
"""

import numpy as np
from scipy import stats
from testbed import NArmedTestbed
from mab import EpsilonGreedyMAB
from simulator import Simulator
from functools import partial
import matplotlib.pyplot as plt

SEED = 1234

num_arms = 10
action_value_mean = 0.0
action_value_std = 1.0
reward_sample_std = 1.0
num_steps = 1000
num_simulations = 1000

EPSILONS = [0, 0.01, 0.1]

# Define the Testbed
Testbed = partial(
	NArmedTestbed,
	num_arms=num_arms, 
	action_value_mean=action_value_mean, 
	action_value_std=action_value_std, 
	reward_sample_std=reward_sample_std, 
	)

avg_rewards, avg_oaps = dict(), dict()

for epsilon in EPSILONS:
	print('# Epsilon =', epsilon)
	# Define the Bandit
	Badit = partial(
		EpsilonGreedyMAB,
		num_arms=num_arms, 
		epsilon=epsilon, 
		inital_value=0.0, 
		step_size=None
		)

	simulator = Simulator(Badit, Testbed, num_simulations=num_simulations, num_steps=num_steps)
	rewards, optimal_action_picked = simulator.run()

	avg_reward = np.mean(rewards, axis=0)
	avg_oap = np.mean(optimal_action_picked, axis=0) * 100

	avg_rewards[epsilon] = avg_reward
	avg_oaps[epsilon] = avg_oap

plt.subplot(2, 1, 1)
plt.plot(avg_rewards[EPSILONS[0]], 'r', label='eps={}'.format(EPSILONS[0]))
plt.plot(avg_rewards[EPSILONS[1]], 'g', label='eps={}'.format(EPSILONS[1]))
plt.plot(avg_rewards[EPSILONS[2]], 'b', label='eps={}'.format(EPSILONS[2]))
plt.xlabel('Step')
plt.ylabel('Avg Reward')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(avg_oaps[EPSILONS[0]], 'r', label='eps={}'.format(EPSILONS[0]))
plt.plot(avg_oaps[EPSILONS[1]], 'g', label='eps={}'.format(EPSILONS[1]))
plt.plot(avg_oaps[EPSILONS[2]], 'b', label='eps={}'.format(EPSILONS[2]))
plt.xlabel('Step')
plt.ylabel('% Optimal Action Picked')
plt.legend()
plt.show()

