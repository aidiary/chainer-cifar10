#/usr/bin/env bash



for model in CONV_relu; do
    python train_cifar10.py --gpu 0 --model $model --out result_$model
done
