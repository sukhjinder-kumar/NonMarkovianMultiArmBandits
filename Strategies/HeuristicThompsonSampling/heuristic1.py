from Strategies.thompson_sampling import ThompsonSampling
import numpy as np

# Score_i = alpha/(alpha+beta) + c/(alhpa + beta)
# We can either sample alpha/beta (1a) from beta like TS or just use there values (1b)

class Heuristic1a(ThompsonSampling):
    def __init__(self, k, c=1, alpha=1, beta=1):
        super().__init__(k)
        self.name = "Thompson Sampling Heuristic1a"
        self.k = k
        self.c = c
        # Initialize the alpha and beta for each arm
        self.alpha = np.ones(k) * alpha  # Success count for each arm
        self.beta = np.ones(k) * beta    # Failure count for each arm

    def get_action(self):
        # Sample a value from Beta distribution for each arm
        scores = np.random.beta(self.alpha, self.beta) + self.c/(self.alpha + self.beta)
        # Select the arm with the highest sampled value
        return np.argmax(scores)


class Heuristic1b(ThompsonSampling):
    def __init__(self, k, c=1, alpha=1, beta=1):
        super().__init__(k)
        self.name = "Thompson Sampling Heuristic1b"
        self.k = k
        self.c = c
        # Initialize the alpha and beta for each arm
        self.alpha = np.ones(k) * alpha  # Success count for each arm
        self.beta = np.ones(k) * beta    # Failure count for each arm

    def get_action(self):
        # Sample a value from Beta distribution for each arm
        scores = self.alpha/(self.alpha + self.beta) + self.c/(self.alpha + self.beta)
        # Select the arm with the highest sampled value
        return np.argmax(scores)