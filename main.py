import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import random

from Strategies.random_action import RandomAction
from Strategies.greedy_strategy import GreedyStrategy
from Strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from Strategies.thompson_sampling import ThompsonSampling
from Strategies.heuristic_thompson_sampling import HeuristicThompsonSampling
from Strategies.policy_gradient import PolicyGradient
from Strategies.ucb import UCBStrategy

from TestCases.test_environment import bernoulli_test1
from Env.mab import MAB
from Env.store import Store

NUMBER_RUN = 2000
TIME_STEP = 1000

mab = MAB(test_env=bernoulli_test1)  # If you change test_env change best mean at the end!
# mab.visualize()

strategies = []
# strategies.append(RandomAction(k=mab.k))
# strategies.append(GreedyStrategy(k=mab.k, optimistic_initial_guess_reward=True))
# strategies.append(EpsilonGreedyStrategy(k=mab.k, epsilon=0.1, optimistic_initial_guess_reward=False))
# strategies.append(PolicyGradient(k=mab.k, alpha=0.03))
# strategies.append(UCBStrategy(k=mab.k, alpha=3, name="3"))
# strategies.append(UCBStrategy(k=mab.k, alpha=2, name="2"))
# strategies.append(UCBStrategy(k=mab.k, alpha=1, name="1"))
# strategies.append(UCBStrategy(k=mab.k, alpha=0.75, name="0.75"))
strategies.append(UCBStrategy(k=mab.k, alpha=0.5, name="0.5"))
strategies.append(ThompsonSampling(k=mab.k))
strategies.append(HeuristicThompsonSampling(k=mab.k))

strategies_names = [strategy.name for strategy in strategies]
store = Store(strategies_names, num_run=NUMBER_RUN, time_step=TIME_STEP)

for strategy in strategies:
    for run in tqdm(range(NUMBER_RUN), desc=f"{strategy.name}", unit=" #Run"):
        strategy.reset()
        for i in range(TIME_STEP):
            action = strategy.get_action()
            reward = mab.get_reward(action)
            strategy.update_strategy(action, reward)
            store.update_store(strategy.name, run, i, reward, action)

store.calculate_mean_per_time_step()
store.calculate_percentage_optimal_choice(mab.get_optimal_action())
store.calculate_avg_cumm_regret_per_time_step(0.7)

store.combine_visualize_mean_interactive()
store.combine_optimal_choice_interactive()
store.combine_avg_cumm_regret_interactive()