import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MAB():
    def __init__(self, test_env): # Accept test_env object
        """
        Multi-armed bandit class.

        Args:
            test_env: An object from test_environments.py (e.g., BernoulliTestEnvironment, GaussianTestEnvironment)
                      that defines the bandit environment.
        """
        self.test_env = test_env
        self.k = test_env.num_arms # Number of arms from test environment
        self.pdfs = []
        for i in range(self.k):
            reward_distribution_func = test_env.get_reward_distribution(i) # Get reward function from test env
            self.pdfs.append(reward_distribution_func) # Store the function

    def get_reward(self, a: int) -> float:
        """
        Get a reward for selecting arm a

        Args:
            a (int): Action (arm index) to take, in [0, 1, ..., k-1].

        Returns:
            float: Reward sampled from the reward distribution of arm 'a'.
        """
        reward = self.pdfs[a]() # Call the reward distribution function
        return reward

    def get_optimal_action(self):
        """
        Get the optimal action (arm) based on the test environment.

        Returns:
            int: Index of the optimal arm.
        """
        return self.test_env.get_optimal_action() # Delegate to test environment

    def visualize(self):
        """
        Visualize the reward distributions of each arm using violin plots.
        """
        # Number of samples to draw from each distribution
        num_samples = 1000

        # Create a list to store the data
        data = []

        # Sample rewards for each action
        for action in range(self.k):
            samples = [self.pdfs[action]() for _ in range(num_samples)]
            data.extend([(sample, action) for sample in samples])

        # Convert data to a DataFrame
        df = pd.DataFrame(data, columns=['Reward', 'Action'])

        # Create the violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Action', y='Reward', data=df, inner='quartile')

        plt.title('Violin Plot of Rewards for Each Action')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.show()