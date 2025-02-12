class EpsilonGreedyStrategy(Strategy):
    def __init__(self, k, epsilon=0.1, optimistic_initial_guess_reward=True):
        self.name = f"epsilon-{epsilon} Greedy Policy with optimism {optimistic_initial_guess_reward}"
        self.epsilon = epsilon
        self.k = k
        self.n = [0]*self.k  # number of attempts made
        self.optimistic_initial_guess_reward = optimistic_initial_guess_reward
        if self.optimistic_initial_guess_reward:
            self.avg_rewards = [5]*self.k
        else:
            self.avg_rewards = [0]*self.k
            
    def get_action(self):
        u = random.uniform(0,1)
        if u > self.epsilon:  # P(greedy action) = P(u > epsilon) = 1-epsilon
            max_value = max(self.avg_rewards)
            max_indices = [i for i, value in enumerate(self.avg_rewards) if value == max_value]
            return int(random.choice(max_indices))
            # return int(np.argmax(self.avg_rewards))
        else:  # P(random action) = epsilon
            return int(random.choice(range(self.k)))  # pick one at random  
        
    def update_strategy(self, cur_action, cur_reward):
        self.n[cur_action] += 1
        alpha = 1/self.n[cur_action]  # after updating n -> n + 1
        self.avg_rewards[cur_action] = self.avg_rewards[cur_action] + alpha * (cur_reward \
                                                                               - self.avg_rewards[cur_action])
    def reset(self):
        self.n = [0]*self.k
        if self.optimistic_initial_guess_reward:
            self.avg_rewards = [5]*self.k
        else:
            self.avg_rewards = [0]*self.k