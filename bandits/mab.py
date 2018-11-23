import numpy as np
from scipy import stats

class MAB:
	def pick(self):
		"""Pick one arm"""
		pass
	def update(self, arm, reward):
		"""Update the estimated reward for the given arm"""
		pass

class EpsilonGreedyMAB(MAB):
	"""Epsilon-greed Multi Armed Bandit"""

	def __init__(self, num_arms, epsilon=0.1, inital_value=0.0, step_size=None, seed=1234):
		"""
		Arguments
		---------
			num_arms: number of arms
			epsilon: exploration probability
			initial_value: initialization value for the value function
			step_size: step size. If None, use 1/n.
			seed: random seed
		"""
		super().__init__()

		self.num_arms = num_arms
		self.epsilon = epsilon
		self.inital_value = inital_value
		self.values = np.ones(self.num_arms) * inital_value
		self.count = np.zeros_like(self.values)
		self.step_size = step_size
		self.rng = np.random.RandomState(seed)

	def pick(self):
		"""Pick one arm"""
		if self.rng.rand() < self.epsilon:
			# Exploration: Pick one arm randomly
			return self.rng.choice(self.num_arms)
		else:
			# Exploitation: Pick the arm with the highest value
			return np.argmax(self.values)

	def _get_step_size(self, arm):
		if self.step_size is None:
			return 1. / self.count[arm]
		else:
			return self.step_size

	def update(self, arm, reward):
		"""Update the estimated reward for the given arm"""
		assert 0 <= arm < self.num_arms, 'Arm index out of range: {}'.format(arm)

		self.count[arm] += 1
		self.values[arm] += self._get_step_size(arm) * (reward - self.values[arm])
