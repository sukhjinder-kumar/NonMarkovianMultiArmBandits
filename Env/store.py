import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import random

class Store():
    LINEWIDTH = 1
    FIGSIZE = (12, 8)

    def __init__(self, names, num_run, time_step):
        self.names = names
        self.num_run = num_run
        self.time_step = time_step
        self.history = {}  # history[name][i][j] = ith run and jth time step = [reward, action]
        for name in names:
            history_name_strategy = [[[0, 0] for _ in range(self.time_step)] for _ in range(self.num_run)]
            self.history.update({name: history_name_strategy})
        self.mean = {}
        self.choose_optimal_action = {}  # % optimal action choosen
        for name in names:
            mean_name_strategy = [-10]*self.time_step
            self.mean.update({name: mean_name_strategy})
        for name in names:
            name_choose_optimal_action = [0]*self.time_step
            self.choose_optimal_action.update({name: name_choose_optimal_action})

    def update_store(self, name_strategy, i, j, reward, action):
        # i being the run number and j being time step
        self.history[name_strategy][i][j] = [reward, action]
        
    def calculate_mean_per_time_step(self):
        for name in self.names:
            for i in range(self.time_step):
                self.mean[name][i] = sum(row[i][0] for row in self.history[name]) / self.num_run

    def calculate_percentage_optimal_choice(self, optimal_action):
        for name in self.names:
            for i in range(self.time_step):
                self.choose_optimal_action[name][i] = sum((row[i][1]==optimal_action) for row in self.history[name]) / self.num_run * 100
    
    def visualize_mean(self, name, linestyle='-', color='b'):
        x = list(range(self.time_step))
        plt.plot(x, self.mean[name], linestyle=linestyle, color=color, \
                 linewidth=Store.LINEWIDTH, label=name)
        plt.xlabel('Time Step')
        plt.ylabel('Mean reward')
        plt.title(f'Expected mean per time step for: {name}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def combine_visualize_mean(self):
        plt.figure(figsize=Store.FIGSIZE)

        # Define a list of colors
        colors = plt.get_cmap('Set1').colors  # Use a colormap to get a list of colors
    
        for i, (name, data) in enumerate(self.mean.items()):
            x = range(self.time_step)
            plt.plot(x, data, color=colors[i % len(colors)], linestyle='-', linewidth=Store.LINEWIDTH, label=name)
    
        plt.xlabel('Time Step')
        plt.ylabel('Reward')
        plt.title('Avg reward for Each Strategy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_action(self, name, linestyle='-', color='b'):
        x = list(range(self.time_step))
        plt.plot(x, self.choose_optimal_action[name], linestyle=linestyle, color=color, \
                 linewidth=Store.LINEWIDTH, label=name)
        plt.xlabel('Time Step')
        plt.ylabel('% Optimal Action')
        plt.title(f'% Optimal Action choose by : {name}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def combine_optimal_choice(self):
        plt.figure(figsize=Store.FIGSIZE)

        # Define a list of colors
        colors = plt.get_cmap('Set1').colors  # Use a colormap to get a list of colors
    
        for i, (name, data) in enumerate(self.choose_optimal_action.items()):
            x = range(self.time_step)
            plt.plot(x, data, color=colors[i % len(colors)], linestyle='-', linewidth=Store.LINEWIDTH, label=name)
    
        plt.xlabel('Time Step')
        plt.ylabel('% Optimal Action')
        plt.title('% Optimal Action choose by each Strategy')
        plt.legend()
        plt.grid(True)
        plt.show()