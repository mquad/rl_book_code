"""
Code to replicate the Optimistic vs. Realistic MAB of Figure 2.3
"""

import numpy as np
from scipy import stats
from testbed import NArmedTestbed, IndependentRandomWalkCallback
from mab import EpsilonGreedyMAB
from simulator import Simulator
from functools import partial
import matplotlib.pyplot as plt

SEED = 1234

num_arms = 10
action_value_mean = 0.0
action_value_std = 1.0
reward_sample_std = 1.0
num_steps = 10000
num_simulations = 1000

SETTINGS = {
	# name: stepsize
	'sample_avg': None,
	'constant-step': 0.1
}

# Define the Testbed
random_walk_callback = IndependentRandomWalkCallback(
	mean=0.0, std=0.1, change_every_steps=1, seed=SEED)

Testbed = partial(
	NArmedTestbed,
	num_arms=num_arms,
	action_value_init='constant',
	action_value_mean=action_value_mean, 
	reward_sample_std=reward_sample_std, 
	callbacks=[random_walk_callback]
	)

avg_rewards, avg_oaps = dict(), dict()

for setting_name, step_size in SETTINGS.items():
	print('Setting:', setting_name)
	print('Step size =', step_size)

	# Define the Bandit
	Badit = partial(
		EpsilonGreedyMAB,
		num_arms=num_arms, 
		epsilon=0.1, 
		inital_value=0.0, 
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
plt.plot(avg_oaps[list(SETTINGS.keys())[1]], 'g', label=list(SETTINGS.keys())[1])
plt.xlabel('Step')
plt.ylabel('% Optimal Action Picked')
plt.legend()
plt.show()

