# test_environments.py
import random

class BernoulliTestEnvironment:
    def __init__(self, num_arms: int, probabilities: list[float]):
        """
        Bernoulli test environment.

        Args:
            num_arms (int): Number of arms.
            probabilities (list): List of probabilities for each arm (between 0 and 1).
                                  Length should be equal to num_arms.
        """
        if len(probabilities) != num_arms:
            raise ValueError("Length of probabilities must be equal to num_arms")
        self.num_arms = num_arms
        self.probabilities = probabilities

    def get_reward_distribution(self, arm_index: int):
        """Returns a function that samples a reward from the Bernoulli distribution for the given arm."""
        p = self.probabilities[arm_index]
        def bernoulli_dist():
            return 1 if random.random() < p else 0
        return bernoulli_dist

    def get_optimal_action(self):
        """Returns the index of the *first* arm with the highest probability (optimal action)."""
        return int(self.probabilities.index(max(self.probabilities))) # Returns first occurrence of max


bernoulli_test1 = BernoulliTestEnvironment(num_arms=3, 
                                           probabilities=[0.2, 0.7, 0.4])
bernoulli_test2 = BernoulliTestEnvironment(num_arms=10, 
                                           probabilities=[0.2, 0.7, 0.4, 0.3, 0.5, 0.65, 0.75, 0.8, 0.69, 0.34])


class GaussianTestEnvironment:
    def __init__(self, num_arms: int, means: list[float], std_devs: list[float]):
        """
        Gaussian test environment.

        Args:
            num_arms (int): Number of arms.
            means (list): List of means for each arm. Length should be equal to num_arms.
            std_dev (float, optional): Standard deviation for all arms. Defaults to 1.
        """
        if len(means) != num_arms or len(std_devs) != num_arms:
            raise ValueError("Length of means must be equal to num_arms")
        self.num_arms = num_arms
        self.means = means
        self.std_devs = std_devs

    def get_reward_distribution(self, arm_index: int):
        """Returns a function that samples a reward from the Gaussian distribution for the given arm."""
        mu = self.means[arm_index]
        sigma = self.std_devs[arm_index]
        def gaussian_dist():
            return random.gauss(mu, sigma)
        return gaussian_dist

    def get_optimal_action(self) -> int:
        """Returns the index of the *first* arm with the highest mean (optimal action)."""
        return int(self.means.index(max(self.means))) # Returns first occurrence of max


guassian_with_uni_variance_test1 = GaussianTestEnvironment(num_arms=4, 
                                                           means=[0.1, -0.2, 0.5, -0.3], 
                                                           std_devs=4*[1])

# You can add more test environments here, e.g., for Kurtosis, etc.
# For simplicity, Kurtosis is not a distribution type itself but a property of distributions.
# If you want to test distributions with different kurtosis, you might want to explore
# other distributions like Laplace (high kurtosis) or Uniform (low kurtosis) and add
# classes for them, or parameterize GaussianTestEnvironment to adjust kurtosis if possible (though Gaussian kurtosis is fixed).

# Example usage within test_environments.py (for demonstration)
if __name__ == '__main__':
    # Bernoulli Test
    print("Bernoulli Environment - Number of Arms:", bernoulli_test1.num_arms)
    print("Bernoulli Environment - Optimal Action:", bernoulli_test1.get_optimal_action())
    print("bernoulli environment - reward sample from arm 1:", bernoulli_test1.get_reward_distribution(1)())

    # Gaussian Test
    print("\nGaussian Environment - Number of Arms:", guassian_with_uni_variance_test1.num_arms)
    print("Gaussian Environment - Optimal Action:", guassian_with_uni_variance_test1.get_optimal_action())
    print("Gaussian Environment - Reward Sample from arm 2:", guassian_with_uni_variance_test1.get_reward_distribution(2)())