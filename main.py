import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import random

from Strategies.random_action import RandomAction
from Strategies.greedy_strategy import GreedyStrategy

from TestCases.test_environment import bernoulli_test1
from Env.mab import MAB
from Env.store import Store

NUMBER_RUN = 20
TIME_STEP = 100

mab = MAB(test_env=bernoulli_test1)
mab.visualize()

strategies = [RandomAction(k=mab.k)]
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

store.combine_visualize_mean()
store.combine_optimal_choice()