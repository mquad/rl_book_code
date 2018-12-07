"""
Code to replicate the examples of Figure 2.4
"""

import numpy as np
from scipy import stats
from testbed import NArmedTestbed
from mab import GradientBasedMAB
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

# Define the Testbed
Testbed = partial(
	NArmedTestbed,
	num_arms=num_arms, 
	action_value_mean=action_value_mean, 
	action_value_std=action_value_std, 
	reward_sample_std=reward_sample_std, 
	)

avg_rewards, avg_oaps = dict(), dict()

Bandits = {
	'GB_0.1_nobase': partial(
		GradientBasedMAB,
		num_arms=num_arms, 
		alpha=0.1,
		use_baseline=False
		),
	'GB_0.1_base': partial(
		GradientBasedMAB,
		num_arms=num_arms, 
		alpha=0.1,
		use_baseline=True
		),
	'GB_0.4_nobase': partial(
		GradientBasedMAB,
		num_arms=num_arms, 
		alpha=0.4,
		use_baseline=False
		),
	'GB_0.4_base': partial(
		GradientBasedMAB,
		num_arms=num_arms, 
		alpha=0.4,
		use_baseline=True
		),
}


for bandit_name, BanditCls in Bandits.items():
	print('Bandit:', bandit_name)

	simulator = Simulator(BanditCls, Testbed, num_simulations=num_simulations, num_steps=num_steps)
	rewards, optimal_action_picked = simulator.run()

	avg_reward = np.mean(rewards, axis=0)
	avg_oap = np.mean(optimal_action_picked, axis=0) * 100

	avg_rewards[bandit_name] = avg_reward
	avg_oaps[bandit_name] = avg_oap

plt.subplot(2, 1, 1)
for bandit_name in Bandits.keys():
	plt.plot(avg_rewards[bandit_name], label=bandit_name)
plt.xlabel('Step')
plt.ylabel('Avg Reward')
plt.legend()


plt.subplot(2, 1, 2)
for bandit_name in Bandits.keys():
	plt.plot(avg_oaps[bandit_name], label=bandit_name)
plt.xlabel('Step')
plt.ylabel('% Optimal Action Picked')
plt.legend()
plt.show()


