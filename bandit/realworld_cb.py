import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
import pandas as pd
from bandit.cb import CB




def one_hot(df, cols):
  """Returns one-hot encoding of DataFrame df including columns in cols."""
  for col in cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)
  return df

def safe_std(values):
  """Remove zero std values for ones."""
  return np.array([val if val != 0.0 else 1.0 for val in values])


def classification_to_bandit_problem(contexts, labels, num_actions=None):
  """Normalize contexts and encode deterministic rewards."""

  if num_actions is None:
    num_actions = np.max(labels) + 1
  num_contexts = contexts.shape[0]

  # Due to random subsampling in small problems, some features may be constant
  sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

  # Normalize features
  contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

  # One hot encode labels as rewards
  rewards = np.zeros((num_contexts, num_actions))
  rewards[np.arange(num_contexts), labels] = 1.0

  return contexts, rewards, (np.ones(num_contexts), labels)


def sample_adult_data(file_name, num_contexts, shuffle_rows=True, remove_underrepresented=False):
  """Returns bandit problem dataset based on the UCI adult data.
  Args:
    file_name: Route of file containing the Adult dataset.
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    remove_underrepresented: If True, removes arms with very few rewards.
  Returns:
    dataset: Sampled matrix with rows: (context, action rewards).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
  Preprocessing:
    * drop rows with missing values
    * convert categorical variables to 1 hot encoding
  https://archive.ics.uci.edu/ml/datasets/census+income
  """
  with tf.io.gfile.GFile(file_name, 'r') as f:
    df = pd.read_csv(f, header=None,
                     na_values=[' ?']).dropna()

  num_actions = 14

  if shuffle_rows:
    df = df.sample(frac=1)
  labels = df[6].astype('category').cat.codes.values
  df = df.drop([6], axis=1)

  # Convert categorical variables to 1 hot encoding
  cols_to_transform = [1, 3, 5, 7, 8, 9, 13, 14]
  df = pd.get_dummies(df, columns=cols_to_transform)

  if remove_underrepresented:
    df, labels = remove_underrepresented_classes(df, labels)
  contexts = df.values

  return classification_to_bandit_problem(contexts, labels, num_actions)




def sample_mushroom_data(file_name, num_contexts, r_noeat=0, r_eat_safe=5, r_eat_poison_bad=-35, r_eat_poison_good=5, prob_poison_bad=0.5):
  """Samples bandit game from Mushroom UCI Dataset.

  Args:
    file_name: Route of file containing the original Mushroom UCI dataset.
    num_contexts: Number of points to sample, i.e. (context, action rewards).
    r_noeat: Reward for not eating a mushroom.
    r_eat_safe: Reward for eating a non-poisonous mushroom.
    r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
    r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
    prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.

  Returns:
    dataset: Sampled matrix with n rows: (context, eat_reward, no_eat_reward).
    opt_vals: Vector of expected optimal (reward, action) for each context.

  We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.
  """

  # first two cols of df encode whether mushroom is edible or poisonous
  df = pd.read_csv(file_name, header=None)
  df = one_hot(df, df.columns)
  ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)

  contexts = df.iloc[ind, 2:]
  no_eat_reward = r_noeat * np.ones((num_contexts, 1))
  random_poison = np.random.choice(
      [r_eat_poison_bad, r_eat_poison_good],
      p=[prob_poison_bad, 1 - prob_poison_bad],
      size=num_contexts)
  eat_reward = r_eat_safe * df.iloc[ind, 0]
  eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
  eat_reward = eat_reward.values.reshape((num_contexts, 1))

  # compute optimal expected reward and optimal actions
  exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
  exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
  opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(
      r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

  if r_noeat > exp_eat_poison_reward:
    # actions: no eat = 0 ; eat = 1
    opt_actions = df.iloc[ind, 0]  # indicator of edible
  else:
    # should always eat (higher expected reward)
    opt_actions = np.ones((num_contexts, 1))

  return contexts, no_eat_reward, eat_reward, opt_exp_reward.values, opt_actions.values



def sample_census_data(file_name, num_contexts, shuffle_rows=True, remove_underrepresented=False):
  """Returns bandit problem dataset based on the UCI census data.
  Args:
    file_name: Route of file containing the Census dataset.
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    remove_underrepresented: If True, removes arms with very few rewards.
  Returns:
    dataset: Sampled matrix with rows: (context, action rewards).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
  Preprocessing:
    * drop rows with missing labels
    * convert categorical variables to 1 hot encoding
  Note: this is the processed (not the 'raw') dataset. It contains a subset
  of the raw features and they've all been discretized.
  https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29
  """
  # Note: this dataset is quite large. It will be slow to load and preprocess.
  with tf.io.gfile.GFile(file_name, 'r') as f:
    df = (pd.read_csv(f, header=0, na_values=['?'])
          .dropna())

  num_actions = 9

  if shuffle_rows:
    df = df.sample(frac=1)
  df = df.iloc[:, :]

  # Assuming what the paper calls response variable is the label?
  labels = df['dOccup'].astype('category').cat.codes.values
  # In addition to label, also drop the (unique?) key.
  df = df.drop(['dOccup', 'caseid'], axis=1)

  # All columns are categorical. Convert to 1 hot encoding.
  df = pd.get_dummies(df, columns=df.columns)

  if remove_underrepresented:
    df, labels = remove_underrepresented_classes(df, labels)
  contexts = df.values
  print('context shape',contexts.shape)

  return classification_to_bandit_problem(contexts, labels, num_actions)



def sample_stock_data(file_name, context_dim, num_actions, num_contexts, sigma, shuffle_rows=True):
  """Samples linear bandit game from stock prices dataset.

  Args:
    file_name: Route of file containing the stock prices dataset.
    context_dim: Context dimension (i.e. vector with the price of each stock).
    num_actions: Number of actions (different linear portfolio strategies).
    num_contexts: Number of contexts to sample.
    sigma: Vector with additive noise levels for each action.
    shuffle_rows: If True, rows from original dataset are shuffled.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_k).
    opt_vals: Vector of expected optimal (reward, action) for each context.
  """

  with tf.io.gfile.GFile(file_name, 'r') as f:
    contexts = np.loadtxt(f, skiprows=1)

  if shuffle_rows:
    np.random.shuffle(contexts)
  contexts = contexts[:num_contexts, :]

  from sklearn.preprocessing import normalize
  #contexts = normalize(contexts)
  betas = np.random.uniform(-1, 1, (context_dim, num_actions))
  betas /= np.linalg.norm(betas, axis=0)

  mean_rewards = np.dot(contexts, betas)
  noise = np.random.normal(scale=sigma, size=mean_rewards.shape)
  rewards = mean_rewards + noise

  opt_actions = np.argmax(mean_rewards, axis=1)
  opt_rewards = [mean_rewards[i, a] for i, a in enumerate(opt_actions)]
  return contexts, rewards, np.array(opt_rewards), opt_actions


def sample_jester_data(file_name, context_dim, num_actions, num_contexts,  shuffle_rows=True, shuffle_cols=False):
  """Samples bandit game from (user, joke) dense subset of Jester dataset.

  Args:
    file_name: Route of file containing the modified Jester dataset.
    context_dim: Context dimension (i.e. vector with some ratings from a user).
    num_actions: Number of actions (number of joke ratings to predict).
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    shuffle_cols: Whether or not context/action jokes are randomly shuffled.

  Returns:
    dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
  """

  with tf.io.gfile.GFile(file_name, 'rb') as f:
    dataset = np.load(f)

  if shuffle_cols:
    dataset = dataset[:, np.random.permutation(dataset.shape[1])]
  if shuffle_rows:
    np.random.shuffle(dataset)
  dataset = dataset[:num_contexts, :]

  assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'

  opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
  opt_rewards = np.array([dataset[i, context_dim + a]
                          for i, a in enumerate(opt_actions)])

  return dataset[:, :context_dim], dataset[:, context_dim:], opt_rewards, opt_actions




class RealWorldCB(CB):
    def __init__(self,
                 name,
                 n_arms,
                 contexts_shape,
                 reward_noise_scale,
                 mode='dbbs',
                 dataset_name='',
                 dataset_path='',
                ):
        self.mode = mode
        self.reward_noise_scale = reward_noise_scale
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.n_arms = n_arms
        self.reset()
        super(RealWorldCB, self).__init__(name,
                         n_arms,
                         contexts_shape)
        
    def reset(self, noise_var = 0.):

        from sklearn.preprocessing import normalize
        if self.mode == 'dbbs':
            all_data = np.load(self.dataset_path, allow_pickle=True)
            context_dim = all_data.item().get('context_dim')
            self.n_arms = all_data.item().get('num_actions')
            self.contexts, self.rewards, _ = np.hsplit(all_data.item().get('dataset'), [context_dim, context_dim + self.n_arms])
            self.contexts = normalize(self.contexts)
            self.best_actions_oracle = all_data.item().get('opt_actions')
            self.true_rewards = all_data.item().get('true_rewards')
            self.best_rewards_oracle = np.array([self.true_rewards[i, self.best_actions_oracle[i]] for i in range(self.rewards.shape[0])])
            
        elif self.mode == 'neuralucb':
            if self.dataset_name.startswith('covertype'):
                self.contexts, best_arm = fetch_openml('covertype', version=3, return_X_y=True)
            elif self.dataset_name.startswith('statlog'):
                self.contexts, best_arm = fetch_openml('shuttle', version=1, return_X_y=True)
                # avoid nan, set nan as -1
            self.contexts, best_arm = shuffle(self.contexts, best_arm)
            self.contexts[np.isnan(self.contexts)] = - 1
            self.contexts = normalize(self.contexts)
            best_arm = OrdinalEncoder(
                dtype=np.int).fit_transform(best_arm.reshape((-1, 1))).squeeze()
            self.true_rewards = np.zeros((best_arm.shape[0], np.max(best_arm) + 1))
            for i in range(best_arm.shape[0]):
                self.true_rewards[i][best_arm[i]] = 1

            self.rewards = np.zeros(self.true_rewards.shape)
            for i in range(self.true_rewards.shape[0]):
                for j in range(self.true_rewards.shape[1]):
                    self.rewards[i][j] = self.true_rewards[i][j] + self.noise_scale * np.random.normal() 

            self.best_actions_oracle = np.argmax(self.true_rewards, axis=1)
            self.best_rewards_oracle = np.array([self.true_rewards[i, self.best_actions_oracle[i]] for i in range(self.rewards.shape[0])])
            
        elif self.mode == 'raw':
            if 'covertype' in self.dataset_name or 'statlog' in self.dataset_name or 'adult' in self.dataset_name or 'census' in self.dataset_name:
                if self.dataset_name.startswith('covertype'):
                    temp_contexts, best_arm = fetch_openml('covertype', version=3, return_X_y=True)
                elif self.dataset_name.startswith('statlog'):
                    temp_contexts, best_arm = fetch_openml('shuttle', version=1, return_X_y=True)
                    # avoid nan, set nan as -1
                elif self.dataset_name.startswith('adult'):
                    file_name = 'adult.full'
                    num_contexts = 45222
                    print('./datasets/' + file_name)
                    sampled_vals = sample_adult_data('./datasets/' + file_name, num_contexts, shuffle_rows=True)
                    temp_contexts, rewards, (opt_rewards, best_arm) = sampled_vals
                    print('shape', temp_contexts.shape)
                elif self.dataset_name.startswith('census'):
                    file_name = 'USCensus1990.data.txt'
                    num_contexts = 150000
                    sampled_vals = sample_census_data('./datasets/' + file_name, num_contexts, shuffle_rows=True)
                    temp_contexts, rewards, (opt_rewards, best_arm) = sampled_vals
                temp_contexts, best_arm = shuffle(temp_contexts, best_arm)
                if type(best_arm) is pd.core.series.Series:
                    best_arm = best_arm.to_numpy()
                if type(temp_contexts) is pd.core.frame.DataFrame:
                    temp_contexts = temp_contexts.to_numpy(dtype='float32')

                temp_contexts[np.isnan(temp_contexts)] = - 1

                if 'covertype' in self.dataset_name:
                    max_v = np.max(temp_contexts)
                    min_v = np.min(temp_contexts)
                    temp_contexts = (temp_contexts - min_v) / (max_v - min_v)

                from sklearn.preprocessing import normalize
                self.contexts = normalize(temp_contexts)
                best_arm = OrdinalEncoder(
                    dtype=np.int).fit_transform(best_arm.reshape((-1, 1))).squeeze()
                self.true_rewards = np.zeros((best_arm.shape[0], np.max(best_arm) + 1))
                for i in range(best_arm.shape[0]):
                    self.true_rewards[i][best_arm[i]] = 1
                self.rewards = np.zeros(self.true_rewards.shape)
                for i in range(self.true_rewards.shape[0]):
                    for j in range(self.true_rewards.shape[1]):
                        self.rewards[i][j] = self.true_rewards[i][j]# + self.reward_noise_scale * np.random.normal()


                self.best_actions_oracle = np.argmax(self.true_rewards, axis=1)
                self.best_rewards_oracle = np.array([self.true_rewards[i, self.best_actions_oracle[i]] for i in range(self.rewards.shape[0])])
            elif self.dataset_name.startswith('mushroom'):
                num_contexts = 50000
                temp_contexts, no_eat_reward, eat_reward, opt_exp_reward, best_arm = sample_mushroom_data('./datasets/mushroom.data', num_contexts)
                self.contexts = normalize(temp_contexts)
                self.contexts = copy.deepcopy(temp_contexts).to_numpy()
                #print('contexts', type(temp_contexts.shape), temp_contexts.shape, self.contexts.shape)
                self.true_rewards = np.zeros((best_arm.shape[0], 2))
                self.true_rewards[:, 0] = no_eat_reward.squeeze()
                self.true_rewards[:, 1] = eat_reward.squeeze()
                self.best_rewards_oracle = opt_exp_reward
                self.best_actions_oracle = best_arm
                self.rewards = np.zeros(self.true_rewards.shape)
                for i in range(self.true_rewards.shape[0]):
                    for j in range(self.true_rewards.shape[1]):
                        self.rewards[i][j] = self.true_rewards[i][j] #+ self.reward_noise_scale * np.random.normal()

            elif self.dataset_name.startswith('financial'):
                num_contexts = 3713
                context_dim = 21
                num_actions = 8
                noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
                file_name = './datasets/raw_stock_contexts'
                temp_contexts, self.true_rewards, self.best_rewards_oracle, self.best_actions_oracle = sample_stock_data(file_name, context_dim, num_actions, num_contexts, noise_stds, shuffle_rows=True)
                #self.contexts = normalize(temp_contexts)
                self.contexts = copy.deepcopy(temp_contexts)
                self.rewards = np.zeros(self.true_rewards.shape)
                for i in range(self.true_rewards.shape[0]):
                    for j in range(self.true_rewards.shape[1]):
                        self.rewards[i][j] = self.true_rewards[i][j] #+ self.reward_noise_scale * np.random.normal()

            elif self.dataset_name.startswith('jester'):
                num_actions = 8
                context_dim = 32
                num_contexts = 19181
                file_name = './datasets/jester_data_40jokes_19181users.npy'
                temp_contexts, self.true_rewards, self.best_rewards_oracle, self.best_actions_oracle = sample_jester_data(file_name, context_dim, num_actions, num_contexts, shuffle_rows=True, shuffle_cols=True)
                #self.contexts = normalize(temp_contexts)
                self.contexts = np.array(temp_contexts)
                self.rewards = np.zeros(self.true_rewards.shape)
                for i in range(self.true_rewards.shape[0]):
                    for j in range(self.true_rewards.shape[1]):
                        self.rewards[i][j] = self.true_rewards[i][j] #+ np.random.normal() * self.reward_noise_scale

        elif self.mode == 'cifar10':


            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

            train_loader = torch.utils.data.DataLoader(
                  torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomCrop(32, 4),
                  transforms.ToTensor(),
                  normalize,
              ]), download=True),
              batch_size=4, shuffle=True,
              num_workers=2, pin_memory=True)

            # transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
            #                                         download=True, transform=transform)
            # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
            #                                           shuffle=True, num_workers=2)
            context_list = []
            label_list = []
            loaderiter = iter(train_loader)
            for datas, labels in loaderiter:
                for i, data in enumerate(datas):

                    context_list.append(data.cpu().detach().numpy())
                    label_temp = np.zeros(10)
                    label_temp[labels[i]] += 1.
                    label_list.append(label_temp)
            self.contexts = np.array(context_list)
            self.rewards = np.array(label_list)
            
            self.best_actions_oracle = np.argmax(self.rewards, axis=1)
            self.best_rewards_oracle = np.array([self.rewards[i, self.best_actions_oracle[i]] for i in range(self.contexts.shape[0])])
        else:
            raise NotImplemented('this mode is not implemented')

    def _get_reward(self, iteration, action, expectation=False):
        if expectation:
            if self.dataset_name.endswith('noise'):
                return self.true_rewards[iteration, action]
        return self.rewards[iteration, action]