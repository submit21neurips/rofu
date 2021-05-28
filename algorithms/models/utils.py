import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from algorithms.models.pytorch_resnet_cifar10.resnet import resnet20, resnet32

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def build_model(model_config, device):
    if model_config['type'] == 'mlp':
        model = Model(input_size=model_config['input_size'],
                      hidden_size=model_config['hidden_size'],
                      n_layers=model_config['n_layers'],
                      activation=model_config['activation'],
                      p=model_config['p'],
                      initialization=model_config['initialization'],
                      output_size=model_config['output_size']).to(device)
    elif model_config['type'] == 'resnet20':
        model = resnet20().to(device)
    elif model_config['type'] == 'resnet32':
        model = resnet32().to(device)
    elif model_config['type'] == 'resmlp':
        model = ResMLP(input_size=model_config['input_size'],
                      hidden_size=model_config['hidden_size'],
                      n_layers=model_config['n_layers'],
                      activation=model_config['activation'],
                      p=model_config['p'],
                      initialization=model_config['initialization'],
                      output_size=model_config['output_size']).to(device)
    else:
        raise NotImplemented('Current model is not supported.')
    return model
            

def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv


class Model(nn.Module):
    """Template for fully connected neural network for scalar approximation.
    """
    def __init__(self, 
                 input_size=1, 
                 hidden_size=2,
                 n_layers=1,
                 activation='ReLU',
                 p=0.0,
                 initialization=None,
                 output_size=1,
                ):
        super(Model, self).__init__()
        
        self.n_layers = n_layers
        
        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, output_size, bias=False)]
            if initialization is not None:
                with torch.no_grad():
                    self.layers[-1].weight.copy_(torch.FloatTensor(initialization['weight']))
                    self.layers[-1].bias.copy_(torch.FloatTensor(initialization['bias']))
        else:
            size  = [input_size] + [hidden_size,] * (self.n_layers-1) + [output_size]
            self.layers = [nn.Linear(size[i], size[i+1]) for i in range(self.n_layers)]
            if initialization is not None:
                with torch.no_grad():
                    for i in range(self.n_layers):
                        print(self.layers[i].weight.shape, self.layers[i].bias.shape)
                        self.layers[i].weight.copy_(torch.FloatTensor(initialization['weight'][i]))
                        self.layers[i].bias.copy_(torch.FloatTensor(initialization['bias'][i]))
        self.layers = nn.ModuleList(self.layers)
        
        self.apply(weights_init_)

        # dropout layer
        self.dropout = nn.Dropout(p=p)
        
        # activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))
            
    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.dropout(self.activation(self.layers[i](x)))
        x = self.layers[-1](x)
        return x
    
    def last_layer(self, x):
        for i in range(self.n_layers - 1):
            x = self.dropout(self.activation(self.layers[i](x)))
        return x
    


class ResMLP(nn.Module):
    """Template for fully connected neural network for scalar approximation.
    """
    def __init__(self, 
                 input_size=1, 
                 hidden_size=2,
                 n_layers=1,
                 activation='ReLU',
                 p=0.0,
                 initialization=None,
                 output_size=1,
                ):
        super(ResMLP, self).__init__()
        
        hidden_size = input_size

        assert(input_size == hidden_size)

        self.n_layers = n_layers
        
        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, output_size, bias=False)]
            if initialization is not None:
                with torch.no_grad():
                    self.layers[-1].weight.copy_(torch.FloatTensor(initialization['weight']))
                    self.layers[-1].bias.copy_(torch.FloatTensor(initialization['bias']))
        else:
            size  = [input_size] + [hidden_size,] * (self.n_layers-1) + [output_size]
            self.layers = [nn.Linear(size[i], size[i+1]) for i in range(self.n_layers)]
            self.bns = [nn.BatchNorm1d(hidden_size).cuda() for i in range(self.n_layers)]
            if initialization is not None:
                with torch.no_grad():
                    for i in range(self.n_layers):
                        print(self.layers[i].weight.shape, self.layers[i].bias.shape)
                        self.layers[i].weight.copy_(torch.FloatTensor(initialization['weight'][i]))
                        self.layers[i].bias.copy_(torch.FloatTensor(initialization['bias'][i]))
        self.layers = nn.ModuleList(self.layers)
        
        # dropout layer
        self.dropout = nn.Dropout(p=p)
        
        # activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))
            
    def forward(self, x):
        for i in range(self.n_layers-1):
            #x = self.dropout(self.activation(x+self.layers[i](x)))
            y =  self.layers[i](x)
            if y.shape[0] > 1:
                y = self.bns[i](y)
            y = self.activation(y)
            x = x + y
        x = self.layers[-1](x)
        return x
    
    def last_layer(self, x):
        raise NotImplemented
        for i in range(self.n_layers - 1):
            #x = self.dropout(self.activation(x+self.layers[i](x)))
            x = self.activation(x+self.layers[i](x))
        return x
    