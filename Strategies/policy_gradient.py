from Strategies.strategy_interface import Strategy
import random
import math


class PolicyGradient(Strategy):
    def __init__(self, k, alpha=0.1):
        self.name = f"Policy Gradient with alpha={alpha}"
        self.k = k
        self.preference_lst = [0]*self.k
        self.alpha = alpha
        self.baseline_lst = [0]*self.k  # running average for that state
        self.n = [0]*self.k  # number of attempts made

    def preference_to_probability(self):
        # probability of picking that actions (of selecting ath machine)
        sum_prob = sum([math.exp(preference) for preference in self.preference_lst])
        prob_lst = [math.exp(preference)/sum_prob for preference in self.preference_lst]
        return prob_lst
        
    def get_action(self):
        prob_lst = self.preference_to_probability()
        actions = list(range(self.k))
        return int(random.choices(actions, weights=prob_lst, k=1)[0])

    def update_strategy(self, cur_action, cur_reward):
        prob_lst = self.preference_to_probability()
        # update preference
        for a in range(self.k):  # sampling over actions
            if a == cur_action:
                self.preference_lst[a] = self.preference_lst[a] + \
                                        self.alpha * (cur_reward - self.baseline_lst[a]) * (1-prob_lst[a])
            else:
                self.preference_lst[a] = self.preference_lst[a] - \
                                        self.alpha * (cur_reward - self.baseline_lst[a]) * prob_lst[a]
        # update baseline
        self.n[cur_action] += 1
        lr = 1/self.n[cur_action]  # after updating n -> n + 1
        self.baseline_lst[cur_action] = self.baseline_lst[cur_action] + lr * (cur_reward \
                                                                              - self.baseline_lst[cur_action])

    def reset(self):
        self.preference_lst = [0]*self.k
        self.baseline_lst = [0]*self.k
        self.n = [0]*self.k