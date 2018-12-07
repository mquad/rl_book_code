import numpy as np
from scipy import stats

def softmax(x):
	"""Compute the softmax of x"""
	# Scale x for numerical stability
	scaled_x = x - np.max(x)
	e_x = np.exp(scaled_x)
	return e_x / np.sum(e_x)


class MAB:
	def pick(self):
		"""Pick one arm"""
		pass
	def update(self, arm, reward):
		"""Update the estimated reward for the given arm"""
		assert 0 <= arm < self.num_arms, 'Arm index out of range: {}'.format(arm)
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
		if self.rng.rand() < self.epsilon:
			# Exploration: Pick one arm randomly
			return self.rng.choice(self.num_arms)
		else:
			# Exploitation: Pick the arm with the highest value
			return np.argmax(self.values)

	def _get_step_size(self, arm):
		if self.step_size is None:
			# Running average
			return 1. / self.count[arm]
		else:
			return self.step_size

	def update(self, arm, reward):
		super().update(arm, reward)
		self.count[arm] += 1
		self.values[arm] += self._get_step_size(arm) * (reward - self.values[arm])

class UCBMAB(MAB):
	"""Upper Confidence Bound Multi Armed Bandit"""

	def __init__(self, num_arms, c=1.0, inital_value=0.0, step_size=None, seed=1234):
		"""
		Arguments
		---------
			num_arms: number of arms
			c: exploration coefficient
			initial_value: initialization value for the value function
			step_size: step size. If None, use 1/n.
			seed: random seed
		"""
		super().__init__()

		self.num_arms = num_arms
		self.c = c
		self.inital_value = inital_value
		self.step_size = step_size
		self.rng = np.random.RandomState(seed)
		self.values = np.zeros(num_arms)
		self.count = np.zeros_like(self.values)
		self.t = 1

	def pick(self):
		unpicked = np.nonzero(self.count == 0)[0]
		if len(unpicked) > 0:
			# Pick the first arm that has zero count deterministically
			return unpicked[0]
		else:
			# Compute the UCB
			ucb = self.values + self.c * np.sqrt(np.log(self.t) / self.count)
			# Return the arm with maximum UCB
			return np.argmax(ucb)

	def _get_step_size(self, arm):
		if self.step_size is None:
			# Running average
			return 1. / self.count[arm]
		else:
			return self.step_size

	def update(self, arm, reward):
		super().update(arm, reward)
		self.t += 1
		self.count[arm] += 1
		self.values[arm] += self._get_step_size(arm) * (reward - self.values[arm])

class GradientBasedMAB(MAB):
	"""Gradient Based Multi Armed Bandit"""
	def __init__(self, num_arms, alpha=0.1, use_baseline=True, seed=1234):
		"""
		Arguments
		---------
			num_arms: number of arms
			alpha: the gradient ascent learning rate
			use_baseline: whether to compute avg total reward as baseline to measure arm preference scores
			seed: random seed
		"""
		self.num_arms = num_arms
		self.alpha = alpha
		self.use_baseline = use_baseline
		self.rng = np.random.RandomState(seed)
		self.preference = np.zeros(self.num_arms)
		self.proba = np.ones_like(self.preference) / self.num_arms 	# start with uniform arm probability
		self.total_avg_reward = 0.0
		self.t = 1

	def pick(self):
		# Sample one arm according to self.proba
		return self.rng.choice(self.num_arms, p=self.proba)


	def update(self, arm, reward):
		super().update(arm, reward)
		self.t += 1
		if self.use_baseline:
			# Update the total avg reward
			self.total_avg_reward = self.total_avg_reward + (reward - self.total_avg_reward) / self.t
		# Update arm prefrence scores with gradient ascent
		rdiff = reward - self.total_avg_reward
		# Move in the direction of sing(rdiff) for the picked arm
		self.preference[arm] += self.alpha * rdiff * (1 - self.proba[arm])
		# Move in the opposite direction for any unpicked arm
		not_picked = np.ones(self.num_arms, dtype=np.bool)
		not_picked[arm] = False
		self.preference[not_picked] -= self.alpha * rdiff * self.proba[arm]
		# Update the arm probabilities
		self.proba = softmax(self.preference)




