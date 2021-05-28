import numpy as np
import torch
from algorithms.base import Base

class GIRO(Base):
    def __init__(self,
                 writer,
                 model_config,
                 batch_size=-1,
                 learning_rate=0.01,
                 train_data_epoch=1,
                 shared_parameter=False,
                 loss_type='mse',
                 n_arms=1,
                 fake_number=1,
                ):
        self.fake_number = fake_number
        self.fake_reward_generator = np.random.normal if loss_type == 'mse' else np.random.binomial
        self.fake_reward_parameter = {'loc':0.0, 'scale':1.0} if loss_type == 'mse' else {'n':1, 'p':0.5}
        super().__init__(writer=writer,
                         model_config=model_config,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         train_data_epoch=train_data_epoch,
                         shared_parameter=shared_parameter,
                         loss_type=loss_type,
                         n_arms=n_arms)
        
    def update(self, action, context, reward):
        self.actions_list.append(action)
        if self.shared_parameter:
            self.contexts_list.append(context[action])
        else:
            self.contexts_list.append(context)
        self.rewards_list.append(reward)
        self.iteration += 1
        for _ in range(self.fake_number):
            self.actions_list.append(action)
            if self.shared_parameter:
                self.contexts_list.append(context[action])
            else:
                self.contexts_list.append(context)
            self.rewards_list.append(self.fake_reward_generator(**self.fake_reward_parameter))
        self.actions = np.array(self.actions_list)
        self.contexts = np.array(self.contexts_list)
        self.rewards = np.array(self.rewards_list)
        
    def decision(self, current_contexts):
        self._train_on_data()
        self.model.eval()
        action_values = self.model.forward(torch.FloatTensor(current_contexts.reshape((-1,) + self.context_shape)).to(self.device)).cpu().detach().squeeze().numpy()
        return np.argmax(action_values)