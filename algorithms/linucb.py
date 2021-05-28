import numpy as np
import math
from algorithms.synthetic_base import SyntheticBase

class LinUCB(SyntheticBase):
    def __init__(self,
                 n_arms,
                 n_features,
                 alpha=1.,
                 prior_multiplier=1.,
                ):
        super().__init__(n_arms=n_arms,
                         n_features=n_features,
                         alpha=alpha,
                         prior_multiplier=prior_multiplier)
     
    def decision(self, current_contexts):
        action_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            action_values[i] = np.dot(self.theta, current_contexts[i]) \
            + self.alpha * math.sqrt(np.matmul(np.matmul(current_contexts[i].reshape(1, -1), np.linalg.inv(self.A)), current_contexts[i].reshape(-1, 1)).squeeze())
        return np.argmax(action_values)
