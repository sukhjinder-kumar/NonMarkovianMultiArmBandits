from Strategies.strategy_interface import Strategy
import random

class RandomAction(Strategy):
    def __init__(self, k, name=""):
        self.name = f"RandomAction {name}"
        self.k = k 
        
    def get_action(self) -> int:
        return int(random.choice(range(self.k)))  # pick one at random  
        
    def update_strategy(self, cur_action: int, cur_reward: float) -> None:
        pass
        
    def reset(self) -> None:
        pass
