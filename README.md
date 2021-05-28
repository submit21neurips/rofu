# rofu
# you can use following steps to run the code:

conda create --name test python=3.7

conda activate test

pip install -r requirements.txt

mkdir dataset

cd datasets

wget https://storage.googleapis.com/bandits_datasets/mushroom.data

wget https://storage.googleapis.com/bandits_datasets/USCensus1990.data.txt

wget https://storage.googleapis.com/bandits_datasets/raw_stock_contexts

wget https://storage.googleapis.com/bandits_datasets/jester_data_40jokes_19181users.npy

wget https://storage.googleapis.com/bandits_datasets/adult.full

cd ..

python example_main.py --algorithms 'rofu' --datasets 'mushroom' --n_sim 16 --timestep 20000 --data_mode 'raw' --model_type 'mlp' \
--hidden_size 100 --n_layers 3 --batch_size 512 --dropout 0.0 --learning_rate .001 \
--train_data_epoch 3 --loss_type 'mse' --last_layer_feature 0 --optimizer 'sgd' \
--epsilon 0.00 --train_lr_decay 0.01 \
--fake_number 1 --lr_decay .9995 \
--rofu_epoch 5 --reg_factor_constant 1. \
--pretrain_nn_epochs 20000 --lr_init 0.0005 \
--n_arms 10 --n_features 20
