import numpy as np
import torch
import math
from algorithms.base import Base

class NN(Base):
    def __init__(self,
                 writer,
                 model_config,
                 batch_size=-1,
                 learning_rate=0.01,
                 train_data_epoch=1,
                 shared_parameter=False,
                 loss_type='mse',
                 n_arms=1,
                 pretrain_nn_epochs=100,
                 all_contexts=[],
                 all_rewards=[]
                ):
        self.pretrain_nn_epochs = pretrain_nn_epochs
        self.all_contexts = all_contexts
        self.all_rewards = all_rewards
        super().__init__(writer=writer,
                         model_config=model_config,
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         train_data_epoch=train_data_epoch,
                         shared_parameter=shared_parameter,
                         loss_type=loss_type,
                         n_arms=n_arms)
        self._pretrain_on_data()
        
    def _pretrain_on_data(self):
        print("hahahahahah")
        total_timestep = self.all_contexts.shape[0]
        self.actions = np.tile(np.arange(self.n_arms), total_timestep)
        self.rewards = self.all_rewards.reshape(-1)
        if not self.shared_parameter:
            self.contexts = np.repeat(self.all_contexts, repeats=self.n_arms, axis=0)
        else:
            self.contexts = self.all_contexts.reshape((-1,) + self.all_contexts.shape[2:])
        for epoch in range(self.pretrain_nn_epochs):
            current_loss = self._train_on_data()
            self.writer.add_scalar('Loss/nn_mse', current_loss, epoch)
        
    def decision(self, current_contexts):
        self.model.eval()
        action_values = self.model.forward(torch.FloatTensor(current_contexts.reshape((-1,) + self.context_shape)).to(self.device)).cpu().detach().squeeze().numpy()
        return np.argmax(action_values)