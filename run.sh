#/usr/bin/env bash

# for model in MLP3 LeNet CONV_relu CONV2; do
#     python train_cifar10.py --gpu 0 --model $model --out result_$model
# done

# python train_cifar10.py --gpu 0 --model CONV2 --out result_CONV2
python train_cifar10.py --gpu 0 --model CONV3 --out result_CONV3
