
class Simulator:
	def __init__(self, Bandit, Testbed, num_simulations=1000, num_steps=2000):
		"""
		Arguments
		--------
			Bandit: class of the bandit
			Testbed: class of the testbed
			num_simulations: total number of simulations
			num_steps: number of steps per simulation
		"""
		self.Bandit = Bandit
		self.Testbed = Testbed
		self.num_simulations = num_simulations
		self.num_steps = num_steps

	def run_once(self, seed):
		# Instantiate the testbed
		testbed = self.Testbed(seed=seed)

		# Instantiate the MAB
		bandit = self.Bandit(seed=seed)

		rewards = []
		optimal_action_picked = []
		for step in range(self.num_steps):
			# Pick one arm
			arm = bandit.pick()
			# Sample the reward
			reward = testbed.sample(arms=arm, steps=1)[0]
			# Update the bandit
			bandit.update(arm, reward)
			# Append the reward and wether the best action was picked
			rewards.append(reward)
			optimal_action_picked.append(arm == testbed.optimal_action())
		return rewards, optimal_action_picked

	def run(self):
		"""Runs the simulation
		Return
		------
		A matrix of shape (num_simulations, num_steps) with the rewards collected throughout the simulation
		A binary matrix of shape (num_simulations, num_steps) indicating when the optimal action was picked by the bandit  
		"""
		rewards = []
		optimal_action_picked = []
		for sim_num in range(1, self.num_simulations + 1):
			if sim_num % 100 == 0:
				print('Running simulation n.', sim_num)
			r, oap = self.run_once(seed=sim_num)
			rewards.append(r)
			optimal_action_picked.append(oap)
		return rewards, optimal_action_picked

