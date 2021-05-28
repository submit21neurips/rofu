import numpy as np 
import os

datasets = ['adult', 'census', 'financial', 'jester', 'mushroom']

for data_name in datasets:
    result_dir_name = 'rofu_ncb/' + data_name + '_results/'
    file_names = os.popen('ls ' + result_dir_name + '| grep rofu_10 | grep npy').read().split()
    minimum_regret = 1000000
    minimum_file = ''
    for file_name in file_names:
        f = np.load(result_dir_name + file_name, allow_pickle=True).item().get('regrets_dict').get(data_name).get('rofu')
        print(np.mean(f[:, 1999]), np.std(f[:, 1999]), f[:, 1999])
        if (np.mean(f[:, 1999]) < minimum_regret):
            minimum_regret = np.mean(f[:, 1999])
            minimum_file = file_name
    print(data_name, minimum_regret)