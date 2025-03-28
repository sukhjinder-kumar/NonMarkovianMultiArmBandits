from Strategies.strategy_interface import Strategy
import random
import math
import matplotlib.pyplot as plt


class PolicyGradientArctan(Strategy):
    def __init__(self, k, alpha=0.1):
        self.name = f"Policy Gradient Arctan with alpha={alpha}"
        self.k = k
        self.preference_lst = [0]*self.k
        self.preference_lst_history = [self.preference_lst]
        self.alpha = alpha
        self.baseline_lst = [0]*self.k  # running average for that state
        self.n = [0]*self.k  # number of attempts made

    def preference_to_probability(self):
        # probability of picking that actions (of selecting ath machine)
        sum_prob = sum([math.exp(math.pi/2 * math.tan(preference)) for preference in self.preference_lst])
        prob_lst = [math.exp(math.pi/2 * math.tan(preference))/sum_prob for preference in self.preference_lst]
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
                                        self.alpha * (cur_reward - self.baseline_lst[a]) * (1-prob_lst[a]) \
                                        * 2/math.pi / (1 + (math.pi/2 * math.tan(self.preference_lst[a]))**2)
            else:
                self.preference_lst[a] = self.preference_lst[a] - \
                                        self.alpha * (cur_reward - self.baseline_lst[a]) * prob_lst[a] \
                                        * 2/math.pi / (1 + (math.pi/2 * math.tan(self.preference_lst[a]))**2)
        # update baseline
        self.n[cur_action] += 1
        lr = 1/self.n[cur_action]  # after updating n -> n + 1
        self.baseline_lst[cur_action] = self.baseline_lst[cur_action] + lr * (cur_reward \
                                                                              - self.baseline_lst[cur_action])
        self.preference_lst_history.append(self.preference_lst.copy())

    def reset(self):
        self.preference_lst = [0]*self.k
        self.baseline_lst = [0]*self.k
        self.n = [0]*self.k

    def plot_preference(self):
        preference_per_arm = zip(*self.preference_lst_history)
        time = range(len(self.preference_lst_history))
        for idx, traj in enumerate(preference_per_arm):
            plt.plot(time, traj, label=f"Arm {idx+1}", marker="o")

        # Labels and legend
        plt.xlabel("Time Step")
        plt.ylabel("Values")
        plt.title("Preference of each arm")
        plt.legend()
        plt.grid()

        # Show the plot
        plt.show()
