import numpy as np
import torch
from algorithms.base import Base

class EpsGreedy(Base):
    def __init__(self,
                 writer,
                 model_config,
                 batch_size=-1,
                 learning_rate=0.01,
                 train_data_epoch=1,
                 shared_parameter=False,
                 loss_type='mse',
                 n_arms=1,
                 epsilon=0.05,
                ):
        self.epsilon = epsilon
        super().__init__(writer=writer,
                         model_config=model_config,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         train_data_epoch=train_data_epoch,
                         shared_parameter=shared_parameter,
                         loss_type=loss_type,
                         n_arms=n_arms)
        
    def decision(self, current_contexts):
        self._train_on_data()
        self.model.eval()
        action_values = self.model.forward(torch.FloatTensor(current_contexts.reshape((-1,) + self.context_shape)).to(self.device)).cpu().detach().squeeze().numpy()
        self.action_values = action_values
        self.mean_reward = action_values
        if np.random.random() > self.epsilon:
            return np.argmax(action_values)
        else:
            return np.random.choice(action_values.shape[0])