import numpy as np
import math
import copy
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from bandit.synthetic_cb import SyntheticCB
from bandit.realworld_cb import RealWorldCB
from algorithms.epsgreedy import EpsGreedy
from algorithms.giro import GIRO
from algorithms.nn import NN
from algorithms.rofu import ROFU
from algorithms.linucb import LinUCB
from algorithms.thompson import Thompson
from algorithms.rofu1_2 import ROFU1_2

parser = argparse.ArgumentParser(description='Arguments for ROFU Bandits')
parser.add_argument('--algorithms', nargs='+', default=['rofu'], help='algorithms for testing')
parser.add_argument('--datasets', nargs='+', default=['covertype'], help='dataset name')
parser.add_argument('--n_sim', type=int, default=1, help='number of times of simulation')
parser.add_argument('--timestep', type=int, help='total timestep for bandits')
parser.add_argument('--save_directory', type=str, help='save results directory')
parser.add_argument('--reward_noise_scale', type=float, default=1.)
# for training optimizer
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate of model')
parser.add_argument('--train_lr_decay', type=float, default=0.)
parser.add_argument('--lr_min', type=float, default=0.001)
# for nn algorithms
parser.add_argument('--dataset_directory', type=str, default='rofu_ncb/datasets/', help='specify where the data is saved')
parser.add_argument('--data_mode', type=str, default='dbbs', help='specify how the real world data or synthetic data is generated')
parser.add_argument('--model_type', type=str, default='mlp', help='specify which kind of model is used for predicting reward')
parser.add_argument('--hidden_size', type=int, default=20, help='hidden size of MLP layers')
parser.add_argument('--n_layers', type=int, default=2, help='number of layers of MLP')
parser.add_argument('--batch_size', type=int, default=256,help='batch size for training')
parser.add_argument('--train_data_epoch', type=int, default=1, help='frequency for doing gradient update for mse/ce')
parser.add_argument('--loss_type', type=str, default='mse', help='type of loss')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer type')
parser.add_argument('--last_layer_feature', type=int, default=0, help='use last layer feature to calculate A or not')
# eps greedy
parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon for eps greedy')
# giro
parser.add_argument('--fake_number', type=int, default=1, help='fake number added to buffer for giro')
# rofu
parser.add_argument('--rofu_epoch', type=int, default=3, help='number of times for rofu gradient update')
parser.add_argument('--reg_factor_constant', type=float, default=1.0, help='regularization constant multiplier')
parser.add_argument('--coef', type=float, default=1.)
parser.add_argument('--reuse_optimizer', type=bool, default=False)
parser.add_argument('--lr_decay', type=float, default=0.9995)
parser.add_argument('--lr_init', type=float, default=0.01)
# nn
parser.add_argument('--pretrain_nn_epochs', type=int, default=100, help='number of epochs to train supervised nn')

# for synthetic dataset
parser.add_argument('--n_arms', type=int, default=7, help='number of arms to play')
parser.add_argument('--n_features', type=int, default=9, help='number of features of parameter')
parser.add_argument('--feature', nargs='+', default=[], help='coefficient of linear bandit')
parser.add_argument('--same', type=bool, default=False, help='specify if use the same network for generating and prediction')

args = parser.parse_args()

# TODO: to be added args
noise_std = 1.
alpha = 1.
prior_multiplier = 1.
activation = 'ReLU'
p = args.dropout

def get_dataset(dataset_name):
    # initialize dataset
    if dataset_name.startswith('covertype') or dataset_name.startswith('statlog') or dataset_name.startswith('adult') or dataset_name.startswith('census')\
     or dataset_name.startswith('mushroom') or dataset_name.startswith('financial') or dataset_name.startswith('jester') or dataset_name == 'cifar10':
        if dataset_name.startswith('covertype'):
            n_arms = 7
            contexts_shape = (54,)
            if 'concat' in dataset_name:
                n_arms = 7
                contexts_shape = (54 + 7,)
        elif dataset_name.startswith('statlog'):
            n_arms = 7
            contexts_shape = (9,)
            if 'concat' in dataset_name:
                n_arms = 7
                contexts_shape = (9 + 7,)
        elif dataset_name.startswith('adult'):
            n_arms = 14
            contexts_shape = (94,)
            if 'concat' in dataset_name:
                n_arms = 14
                contexts_shape = (94 + 14,)
        elif dataset_name.startswith('census'):
            n_arms = 9
            contexts_shape = (387,)
            if 'concat' in dataset_name:
                n_arms = 9
                contexts_shape = (387 + 9,)
        elif dataset_name.startswith('mushroom'):
            n_arms = 2
            contexts_shape = (117,) # after one hot preprocessing by data_sampler
            if 'concat' in dataset_name:
                n_arms = 2
                contexts_shape = (117 + 2,)
        elif dataset_name.startswith('financial'):
            n_arms = 8
            contexts_shape = (21,)
            if 'concat' in dataset_name:
                n_arms = 8
                contexts_shape = (21 + 8,)
        elif dataset_name.startswith('jester'):
            n_arms = 8
            contexts_shape = (32,)
            if 'concat' in dataset_name:
                n_arms = 8
                contexts_shape = (32 + 8,)
        elif dataset_name.startswith('wheel'):
            raise NotImplementedError
        else:# dataset_name == 'cifar10':
            n_arms = 10
            contexts_shape = (3, 32, 32)
        dataset = RealWorldCB(reward_noise_scale=args.reward_noise_scale,
                              name=dataset_name,
                              n_arms=n_arms,
                              contexts_shape=contexts_shape,
                              mode=args.data_mode,
                              dataset_name=dataset_name,
                              dataset_path=args.dataset_directory + args.data_mode + dataset_name)
    elif dataset_name == 'linear':
        dataset = SyntheticCB(name=dataset_name,
                              n_arms=args.n_arms,
                              contexts_shape=(args.n_features,),
                              timestep=args.timestep,
                              mode=args.data_mode, # linear for linear, resnet for resnet
                              same=args.same,
                              noise_std=1., # TODO: replace or not
                              feature=np.array(args.feature))
    elif dataset_name == 'resnet':
        dataset = SyntheticCB(name=dataset_name,
                              n_arms=args.n_arms,
                              contexts_shape=(3, 32, 32),
                              timestep=args.timestep,
                              mode=args.data_mode, # linear for linear, resnet for resnet
                              same=args.same,
                              noise_std=noise_std,
                              feature=np.array(args.feature))
    else:
        raise NotImplementedError('This type of dataset is not supported yet.')
    return dataset

def generate_model_config(input_size, output_size):
    if args.model_type == 'mlp':
        return {'type': 'mlp',
               'input_size': input_size[0],
               'hidden_size': args.hidden_size,
               'n_layers': args.n_layers,
               'activation': activation,
               'p': p, 
               'initialization': None,
               'output_size': output_size}
    elif args.model_type == 'resnet':
        return {'type': 'resnet32'}
        if args.same:
            return {'type': 'resnet20'}
        else:
            return {'type': 'resnet32'}
    if args.model_type == 'resmlp':
        return {'type': 'resmlp',
               'input_size': input_size[0],
               'hidden_size': args.hidden_size,
               'n_layers': args.n_layers,
               'activation': activation,
               'p': p, 
               'initialization': None,
               'output_size': output_size}
    else:
        raise NotImplementedError('This kind of model is not supported right now.')

def get_algorithm(algorithm, dataset, writer):
     # initialize algorithm
    shared_parameter = False if dataset.name in ['covertype', 'statlog', 'adult', 'census', 'mushroom', 'financial', 'jester', 'cifar10' ,'resnet',\
     'statlognoise', 'covertypenoise', 'adultnoise', 'censusnoise', 'mushroomnoise', 'financialnoise', 'jesternoise'] else True
    if algorithm == 'linucb':
        alg = LinUCB(args.n_arms,
                     args.n_features,
                     alpha=alpha,
                     prior_multiplier=prior_multiplier)
    elif algorithm == 'thompson':
        alg = Thompson(args.n_arms,
                       args.n_features,
                       alpha=alpha,
                       prior_multiplier=prior_multiplier)
    elif algorithm == 'epsgreedy':
        alg = EpsGreedy(
                 writer,
                 generate_model_config(dataset.contexts_shape, 1 if shared_parameter else dataset.n_arms),
                 batch_size=args.batch_size,
                 learning_rate=args.learning_rate,
                 train_data_epoch=args.train_data_epoch,
                 shared_parameter=shared_parameter,
                 loss_type=args.loss_type,
                 n_arms=dataset.n_arms,
                 epsilon=args.epsilon)
    elif algorithm == 'giro': 
        alg = GIRO(
            writer,
            generate_model_config(dataset.contexts_shape, 1 if shared_parameter else dataset.n_arms),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_data_epoch=args.train_data_epoch,
            shared_parameter=shared_parameter,
            loss_type=args.loss_type,
            n_arms=dataset.n_arms,
            fake_number=args.fake_number)
    elif algorithm == 'nn':
        print("checking shape", dataset.get_all_contexts()[:args.timestep].shape, dataset.get_all_rewards()[:args.timestep].shape)
        alg = NN(
            writer,
            generate_model_config(dataset.contexts_shape, 1 if shared_parameter else dataset.n_arms),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_data_epoch=args.train_data_epoch,
            shared_parameter=shared_parameter,
            loss_type=args.loss_type,
            n_arms=dataset.n_arms,
            pretrain_nn_epochs=args.pretrain_nn_epochs,
            all_contexts = dataset.get_all_contexts()[:args.timestep],
            all_rewards = dataset.get_all_rewards()[:args.timestep])
    elif algorithm == 'rofu':
        assert args.loss_type == 'mse'
        alg = ROFU(
            writer,
            generate_model_config(dataset.contexts_shape, 1 if shared_parameter else dataset.n_arms),
            lr_init=args.lr_init,
            lr_decay=args.lr_decay,
            coef=args.coef,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_lr_decay=args.train_lr_decay,
            train_data_epoch=args.train_data_epoch,
            shared_parameter=shared_parameter,
            loss_type=args.loss_type,
            n_arms=dataset.n_arms,
            rofu_epoch=args.rofu_epoch,
            reg_constant=args.reg_factor_constant,
            last_layer_feature=args.last_layer_feature,
            optimizer_name=args.optimizer)
    elif algorithm == 'rofu1_2':
        alg = ROFU1_2(
            writer,
            generate_model_config(dataset.contexts_shape, 1 if shared_parameter else dataset.n_arms),
            coef=args.coef,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_data_epoch=args.train_data_epoch,
            shared_parameter=shared_parameter,
            loss_type=args.loss_type,
            n_arms=dataset.n_arms,
            rofu_epoch=args.rofu_epoch,
            reg_constant=args.reg_factor_constant,
            last_layer_feature=args.last_layer_feature,
            optimizer_name=args.optimizer)

    else:
        raise NotImplementedError('This algorithm is not supported currently.')
    return alg
        
def save_data(sim, regrets_dict, action_value, mean_value, suffix):
    all_colors = ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'pink', 'black']
    for dataset_name in args.datasets:
        #random_suffix = np.random.randint(100000)
        current_path = args.save_directory + dataset_name + f'_results_{suffix}/'
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        all_args = copy.deepcopy(vars(args))
        all_args.pop('save_directory')
        all_args.pop('dataset_directory')
        all_args.pop('datasets')
        current_name = ''
        for key, value in all_args.items():
            if type(value) is int or type(value) is float or type(value) is bool:
                current_name = current_name + str(value) + '_'
            elif type(value) is list:
                current_name = current_name + ','.join(value) + '_'
            elif type(value) is str:
                current_name = current_name + value
        random_end = np.random.randint(0, 100000)
        np.save(current_path + current_name + '.npy', {'regrets_dict': regrets_dict, 'all_args': all_args, 'action_value': np.array(action_value), 'mean_value': np.array(mean_value)})
        fig, ax = plt.subplots(figsize=(6, 4), nrows=1, ncols=1)
        for i, algorithm in enumerate(args.algorithms):
            regrets = np.array(regrets_dict[dataset_name][algorithm][:sim+1])
            mean_regret = np.mean(regrets, axis=0)
            t = np.arange(regrets.shape[-1])
            ax.plot(t, mean_regret, color=all_colors[i], label=algorithm)
            
        ax.set_title('Cumulative regret on ' + dataset_name)
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.xlabel('timestep')
        plt.ylabel('regret')
        fig.savefig(current_path + current_name + '.pdf')


def train():
    random_suffix = np.random.randint(100000)
    args.save_directory = args.save_directory
    regrets_dict = {}
    for dataset_name in args.datasets:
        regrets_dict[dataset_name] = {}
        for algorithm in args.algorithms:
            regrets_dict[dataset_name][algorithm] = np.zeros((args.n_sim, args.timestep))
    writer = SummaryWriter(comment=algorithm+args.datasets[0]+args.model_type)
    action_value = []
    mean_value = []
    for sim in range(args.n_sim):
        for dataset_name in args.datasets:
            dataset = get_dataset(dataset_name)
            for algorithm in args.algorithms:
                current_action_value = []
                current_mean = []
                alg = get_algorithm(algorithm, dataset, writer)
                dataset.reset_iteration()
                current_evaluation_regrets = []
                for t in range(dataset.n_arms * 3):
                    current_contexts = dataset.get_next_context()
                    current_action = t % dataset.n_arms
                    current_reward = dataset.step(current_action)
                    alg.update(current_action, current_contexts, current_reward)
                    alg._train_on_data()
                    current_regret = dataset.current_regret()
                    current_evaluation_regrets.append(current_regret)
                for t in range(args.timestep - dataset.n_arms * 3):
                    #print("info", t)
                    t_bias = dataset.n_arms * 3
                    current_contexts = dataset.get_next_context()

                    current_action = alg.decision(current_contexts)
                    current_reward = dataset.step(current_action)
                    alg.update(current_action, current_contexts, current_reward)
                    current_regret = dataset.current_regret()
                    current_evaluation_regrets.append(current_regret)
                    #current_action_value.append(alg.action_values)
                    #current_mean.append(alg.mean_reward)

                    writer.add_scalar(f"regret_{sim}", np.sum(current_evaluation_regrets), t)
                regrets_dict[dataset_name][algorithm][sim] = np.cumsum(current_evaluation_regrets)
                action_value.append(current_action_value)
                mean_value.append(current_mean)
        save_data(sim, regrets_dict, action_value, mean_value, random_suffix)
        
if __name__ == '__main__':
    train()