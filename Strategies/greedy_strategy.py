from Strategies.strategy_interface import Strategy
import random

class GreedyStrategy(Strategy):
    def __init__(self, k, optimistic_initial_guess_reward=True):
        self.name = f"Greedy with optimism {optimistic_initial_guess_reward}"
        self.k = k
        self.n = [0]*self.k  # number of attempts made
        self.optimistic_initial_guess_reward = optimistic_initial_guess_reward
        if self.optimistic_initial_guess_reward:
            self.avg_rewards = [1000]*self.k
        else:
            self.avg_rewards = [0]*self.k
            
    def get_action(self):
        max_value = max(self.avg_rewards)
        max_indices = [i for i, value in enumerate(self.avg_rewards) if value == max_value]
        return int(random.choice(max_indices))
        # return int(np.argmax(self.avg_rewards))
        
    def update_strategy(self, cur_action, cur_reward):
        self.n[cur_action] += 1
        alpha = 1/self.n[cur_action]  # after updating n -> n + 1!
        self.avg_rewards[cur_action] = self.avg_rewards[cur_action] + alpha * (cur_reward \
                                                                               - self.avg_rewards[cur_action])
    def reset(self):
        self.n = [0]*self.k
        if self.optimistic_initial_guess_reward:
            self.avg_rewards = [1000]*self.k
        else:
            self.avg_rewards = [0]*self.k