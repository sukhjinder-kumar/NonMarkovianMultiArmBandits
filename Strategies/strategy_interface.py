# An interface for all the strategies for picking optimal action
class Strategy():
    def __init__(self, num_arms):
        self.name = "Generic Strategy"
        self.k = num_arms

    def get_action(self) -> int:
        # returns action \in [0, 1, ..., k-1]
        pass

    def update_strategy(self, cur_action: int, cur_reward: float) -> None:
        # update the strategy based on reward in accordance with action you just took
        pass

    def reset(self) -> None:
        # resets strategy state to original
        pass
