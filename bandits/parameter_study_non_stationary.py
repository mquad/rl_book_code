"""
Code to replicate for exercise 2.11
"""

import numpy as np
from scipy import stats
from testbed import NArmedTestbed, IndependentRandomWalkCallback
from mab import EpsilonGreedyMAB, UCBMAB, GradientBasedMAB
from simulator import Simulator
from functools import partial
import matplotlib.pyplot as plt
from collections import defaultdict

def run_sim(Bandit, Testbed, num_simulations, num_steps):
	simulator = Simulator(Bandit, Testbed, num_simulations=num_simulations, num_steps=num_steps)
	rewards, optimal_action_picked = simulator.run()
	# average over the last 100k steps
	return np.mean(rewards[:-100000])

SEED = 1234

num_arms = 10
action_value_mean = 0.0
action_value_std = 1.0
reward_sample_std = 1.0
num_steps = 200
num_simulations = 100

# Define the Testbed
random_walk_callback = IndependentRandomWalkCallback(
	mean=0.0, std=0.1, change_every_steps=1, seed=SEED
	)

Testbed = partial(
	NArmedTestbed,
	num_arms=num_arms,
	action_value_init='constant',
	action_value_mean=action_value_mean, 
	reward_sample_std=reward_sample_std, 
	callbacks=[random_walk_callback]
	)

avg_rewards, avg_oaps = dict(), dict()

# Define the parameter space
eps_greedy__epsilon = np.logspace(-7,-2, base=2, num=6)
greedy_optim__init_value = np.logspace(-2, 2, base=2, num=5)
gradient__alpha = np.logspace(-5, 2, base=2, num=8)
ucb__c = np.logspace(-4, 2, base=2, num=7)

avg_rewards = defaultdict(list)

# Epsilon greedy
for epsilon in eps_greedy__epsilon:
	print('# Epsilon Greedy (Epsilon={})'.format(epsilon))
	# Define the Bandit
	Bandit = partial(
		EpsilonGreedyMAB,
		num_arms=num_arms, 
		epsilon=epsilon, 
		inital_value=0.0,
		step_size=None
		)
	avgr = run_sim(Bandit, Testbed, num_simulations, num_steps)
	avg_rewards['eps_greedy'].append((epsilon, avgr))

# Constant step size epsilon greedy
for epsilon in eps_greedy__epsilon:
	print('# Constnat-step Epsilon Greedy (Epsilon={})'.format(epsilon))
	# Define the Bandit
	Bandit = partial(
		EpsilonGreedyMAB,
		num_arms=num_arms, 
		epsilon=epsilon, 
		inital_value=0.0,
		step_size=0.1
		)
	avgr = run_sim(Bandit, Testbed, num_simulations, num_steps)
	avg_rewards['const_step_eps_greedy'].append((epsilon, avgr))

# Greedy - Optimistic initialization
for initial_value in greedy_optim__init_value:
	print('# Optimistic Greedy (initial_value={})'.format(initial_value))
	# Define the Bandit
	Bandit = partial(
		EpsilonGreedyMAB,
		num_arms=num_arms, 
		epsilon=0, 
		inital_value=initial_value,
		step_size=None
		)
	avgr = run_sim(Bandit, Testbed, num_simulations, num_steps)
	avg_rewards['greedy_optim'].append((initial_value, avgr))

# Gradient based
for alpha in gradient__alpha:
	print("# Gradient Based (alpha={})".format(alpha))
	Bandit = partial(
		GradientBasedMAB,
		num_arms=num_arms, 
		alpha=alpha,
		use_baseline=True
		)
	avgr = run_sim(Bandit, Testbed, num_simulations, num_steps)
	avg_rewards['gradient_based'].append((alpha, avgr))

# UCB
for c in ucb__c:
	print("# UCB (c={})".format(c))
	Bandit = partial(
		UCBMAB,
		num_arms=num_arms, 
		c=c, 
		inital_value=0.0, 
		step_size=None
		)
	avgr = run_sim(Bandit, Testbed, num_simulations, num_steps)
	avg_rewards['UCB'].append((c, avgr))

print(avg_rewards)
for bandit_name, results in avg_rewards.items():
	x, y = list(zip(*results))
	plt.plot(x, y, label=bandit_name)

plt.xscale('log')
plt.xlabel('Parameter')
plt.ylabel('Avg Reward')
plt.legend()
#plt.show()
plt.savefig('figures/parameter_study_non_stationary.png')
