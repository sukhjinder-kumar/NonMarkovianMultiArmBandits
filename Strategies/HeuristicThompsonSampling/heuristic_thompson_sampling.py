from Strategies.strategy_interface import Strategy
import numpy as np
from scipy.stats import skew


class HeuristicThompsonSampling(Strategy):
    def __init__(self, k, alpha=1, beta=1, lambda_fisher=1.0, lambda_skew=1.0, lambda_ucb=1.0):
        super().__init__(k)
        self.name = "Heuristic Thompson Sampling"
        self.k = k
        self.alpha = np.ones(k) * alpha
        self.beta = np.ones(k) * beta
        self.lambda_fisher = lambda_fisher
        self.lambda_skew = lambda_skew
        self.lambda_ucb = lambda_ucb
        self.n = np.zeros(k)  # Number of pulls for each arm

    def compute_fisher_information(self, arm):
        # For Beta distribution, Fisher Information is calculated as:
        return (self.alpha[arm] * self.beta[arm]) / ((self.alpha[arm] + self.beta[arm]) ** 2 * (self.alpha[arm] + self.beta[arm] + 1))

    def compute_skewness(self, arm):
        # Skewness is a measure of asymmetry of the distribution (could use a different heuristic)
        # Using the skew from the Beta distribution approximation here as an example.
        return skew([np.random.beta(self.alpha[arm], self.beta[arm]) for _ in range(100)])

    def compute_ucb(self, arm, t):
        # UCB approach (upper confidence bound)
        return (self.alpha[arm] / (self.alpha[arm] + self.beta[arm])) + np.sqrt(2 * np.log(t) / (self.n[arm] + 1))

    def get_action(self):
        scores = np.zeros(self.k)
        for arm in range(self.k):
            # Compute the exploitation term (mean reward)
            mean_reward = self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
            
            # Compute the heuristics
            fisher_info = self.compute_fisher_information(arm)
            skew_val = self.compute_skewness(arm)
            ucb_val = self.compute_ucb(arm, self.n[arm])
            
            # Combine all heuristics into a score
            scores[arm] = mean_reward + self.lambda_fisher * fisher_info - self.lambda_skew * skew_val + self.lambda_ucb * ucb_val
        
        # Choose the arm with the highest combined score
        return np.argmax(scores)

    def update_strategy(self, cur_action, cur_reward):
        # Update based on reward received
        if cur_reward > 0:
            self.alpha[cur_action] += 1
        else:
            self.beta[cur_action] += 1
        self.n[cur_action] += 1

    def reset(self):
        # Reset all parameters
        self.alpha = np.ones(self.k)
        self.beta = np.ones(self.k)
        self.n = np.zeros(self.k)