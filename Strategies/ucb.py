from Strategies.strategy_interface import Strategy
import numpy as np


# upper confidence bound algorithm
class UCBStrategy(Strategy):
    def __init__(self, k, alpha, name=""):
        self.k = k
        self.name = "Upper Confidence Bound (UCB)" + name
        self.alpha = alpha  # Exploration parameter
        self.counts = np.zeros(k, dtype=int)  # Number of times each action was selected
        self.avg_rewards = np.zeros(k, dtype=float)  # Estimated reward values

    def get_action(self):
        total_counts = np.sum(self.counts)  # equivalently this is the current time
        
        if total_counts < self.k:  
            # Play each action at least once (sequentially till all have one)
            return total_counts
        
        # Calculate UCB values
        ucb_values = self.avg_rewards + self.alpha * np.sqrt(np.log(total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update_strategy(self, cur_action, cur_reward):
        self.counts[cur_action] += 1
        n = self.counts[cur_action]
        self.avg_rewards[cur_action] += (cur_reward - self.avg_rewards[cur_action]) / n  # Incremental mean update

    def reset(self):
        self.counts.fill(0)
        self.avg_rewards.fill(0)