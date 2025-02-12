from Strategies.strategy_interface import Strategy
import numpy as np

class ThompsonSampling(Strategy):
    def __init__(self, k, alpha=1, beta=1):
        super().__init__(k)
        self.name = "Thompson Sampling"
        self.k = k
        # Initialize the alpha and beta for each arm
        self.alpha = np.ones(k) * alpha  # Success count for each arm
        self.beta = np.ones(k) * beta    # Failure count for each arm

    def get_action(self):
        # Sample a value from Beta distribution for each arm
        theta_samples = np.random.beta(self.alpha, self.beta)
        # Select the arm with the highest sampled value
        return np.argmax(theta_samples)

    def update_strategy(self, cur_action, cur_reward):
        # If the reward is positive, increment the success count for the chosen arm
        if cur_reward > 0:
            self.alpha[cur_action] += 1
        # If the reward is negative, increment the failure count for the chosen arm
        else:
            self.beta[cur_action] += 1

    def reset(self):
        # Reset the alpha and beta values to the initial values
        self.alpha = np.ones(self.k)
        self.beta = np.ones(self.k)