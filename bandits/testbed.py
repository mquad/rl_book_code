import numpy as np
import matplotlib.pyplot as plt

class NArmedTestbed:
	"""Testbed for multi-armed bandits with Normally distributed rewards. 
	Arm's action values are normally sampled too."""

	def __init__(
		self, 
		num_arms,
		action_value_init='normal', 
		action_value_mean=0.0, 
		action_value_std=1.0, 
		reward_sample_std=1.0, 
		seed=1234, 
		callbacks=None):
		"""
		Arguments
		---------
			num_arms: number of arms
			action_value_init: initialization of the action values (normal, constant)
			action_value_mean: mean of the true action value
			action_value_std: std of the true action value
			reward_sample_std: std of the sampled rewards
			seed: random seed
			callbacks: list of functions to be called after call to sample() to change the arm reward distributions.
				Functions will be applied in order.
		"""
		assert num_arms > 1
		assert action_value_init in {'normal', 'constant'}

		self.rng = np.random.RandomState(seed)
		self.num_arms = num_arms
		self.action_value_init = action_value_init
		self.action_value_mean = action_value_mean
		self.action_value_std = action_value_std
		self.reward_sample_std = reward_sample_std
		self.callbacks = callbacks
		self._init_values()
		self._init_reward_dist()

	def _init_values(self):
		if self.action_value_init == 'normal':
			# Sample normally distributed true action values for each arm 
			self.arm_values = self.rng.normal(self.action_value_mean, self.action_value_std, size=self.num_arms)
		if self.action_value_init == 'constant':
			self.arm_values = np.ones(self.num_arms) * self.action_value_mean

	def _init_reward_dist(self):
		# Set normally distributed rewards around each arm's value (mean, std)
		# self.arm_reward_dist = [stats.norm(av, self.reward_sample_std) for av in self.arm_values]
		self.arm_reward_dist = [(av, self.reward_sample_std) for av in self.arm_values]

	def optimal_action(self):
		"""Return the index of the optimal action in the testbed"""
		return np.argmax(self.arm_values)

	def sample(self, steps=1, arms=None):
		"""Sample rewards from each arm
		Arguments
		---------
			steps: the number of sampling steps
			arm: lst of indices of the arm(s) to sample
		"""
		if arms is None:
			indices = range(self.num_arms)
		elif not isinstance(arms, list):
			indices = [arms]
		else:
			indices = arms
		dists = [self.arm_reward_dist[i] for i in indices]
		# Sample the rewards
		# samples = [ard.rvs(steps) for ard in dists]
		samples = [self.rng.normal(dmean, dstd, size=steps) for dmean, dstd in dists]

		# Apply the callbacks in order
		if self.callbacks:
			for callback in self.callbacks:
				self.arm_values = callback.call(self.arm_values)
			self._init_reward_dist()
		return samples

	def plot_distribution(self, num_samples=1000):
		"""Plots each arm's distribution
		Arguments
		---------
			num_samples: the number of rewards sampled per arm
		"""
		# Sample rewards for all arms
		sampled_rewards = self.sample(num_samples)
		# Plot the distributions
		pos = np.arange(self.num_arms)
		plt.violinplot(sampled_rewards, pos)
		plt.axhline(self.action_value_mean, linestyle='--')
		plt.xticks(pos)
		plt.xlabel('Arm')
		plt.ylabel('Reward distribution')
		plt.show()

class IndependentRandomWalkCallback:
	"""Add an independent increment to action value means to simulate non-stationarity"""
	def __init__(self, mean=0.00, std=0.01, change_every_steps=1, seed=1234):
		assert change_every_steps > 0
		self.mean = mean
		self.std = std
		self.change_every_steps = change_every_steps
		self.steps_to_next_change = self.change_every_steps
		self.rng = np.random.RandomState(seed)

	def call(self, arm_values):
		if self.steps_to_next_change == 1:
			# Add an independent random increment to the action value means
			arm_value_delta = self.rng.normal(self.mean, self.std, size=len(arm_values))
			new_arm_values = arm_values + arm_value_delta
			# Reset the counter
			self.steps_to_next_change = self.change_every_steps
		else:
			self.steps_to_next_change -= 1
		return new_arm_values


if __name__ == '__main__':
	testbed = NArmedTestbed(
		num_arms=10, action_value_init='normal', action_value_mean=1.0, action_value_std=1.0, reward_sample_std=.5)
	testbed.plot_distribution()
	testbed = NArmedTestbed(
		num_arms=10, action_value_init='constant', action_value_mean=1.0, reward_sample_std=.5)
	testbed.plot_distribution()

	random_walk_callback = IndependentRandomWalkCallback(
		mean=0.0, std=0.1, change_every_steps=1, seed=1234)
	testbed = NArmedTestbed(
		num_arms=10, action_value_init='constant', action_value_mean=1.0, reward_sample_std=.5, callbacks=[random_walk_callback])
	testbed.plot_distribution(num_samples=10000)
