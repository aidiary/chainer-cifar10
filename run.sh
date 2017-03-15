#/usr/bin/env bash

# for model in MLP3 LeNet CONV_relu CONV2; do
#     python train_cifar10.py --gpu 0 --model $model --out result_$model
# done

# MomentumSGDの学習率を0.1から0.05に下げた
python train_cifar10.py --gpu 0 --model CONV3 --out result_CONV3
