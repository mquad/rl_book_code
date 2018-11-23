"""
Code to replicate the Optimistic vs. Realistic MAB of Figure 2.3
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
step_size = 0.1

SETTINGS = {
	# name: (initial_value, epsilon)
	'optimistic (greedy)': (5.0, 0),
	'realistic (eps=0.1)': (0.0, 0.1)
}

# Define the Testbed
Testbed = partial(
	NArmedTestbed,
	num_arms=num_arms, 
	action_value_mean=action_value_mean, 
	action_value_std=action_value_std, 
	reward_sample_std=reward_sample_std, 
	)

avg_rewards, avg_oaps = dict(), dict()

for setting_name, (init_value, epsilon) in SETTINGS.items():
	print('Setting:', setting_name)
	print('Initial value =', init_value)
	print('Epsilon =', epsilon)

	# Define the Bandit
	Badit = partial(
		EpsilonGreedyMAB,
		num_arms=num_arms, 
		epsilon=epsilon, 
		inital_value=init_value, 
		step_size=step_size
		)

	simulator = Simulator(Badit, Testbed, num_simulations=num_simulations, num_steps=num_steps)
	rewards, optimal_action_picked = simulator.run()

	avg_reward = np.mean(rewards, axis=0)
	avg_oap = np.mean(optimal_action_picked, axis=0) * 100

	avg_rewards[setting_name] = avg_reward
	avg_oaps[setting_name] = avg_oap

plt.subplot(2, 1, 1)
plt.plot(avg_rewards[list(SETTINGS.keys())[0]], 'r', label=list(SETTINGS.keys())[0])
plt.plot(avg_rewards[list(SETTINGS.keys())[1]], 'g', label=list(SETTINGS.keys())[1])
plt.xlabel('Step')
plt.ylabel('Avg Reward')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(avg_oaps[list(SETTINGS.keys())[0]], 'r', label=list(SETTINGS.keys())[0])
plt.plot(avg_oaps[list(SETTINGS.keys())[1]], 'g', label=list(SETTINGS.keys())[0])
plt.xlabel('Step')
plt.ylabel('% Optimal Action Picked')
plt.legend()
plt.show()

