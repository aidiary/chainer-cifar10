import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from net import *


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR-10 example:')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--model', '-m',
                        choices=('MLP3', 'LeNet', 'CONV_relu', 'CONV2', 'CONV3'),
                        default='LeNet', help='model type')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('model type: {}'.format(args.model))
    print('')

    # load CIFAR-10 dataset
    class_labels = 10
    train, test = chainer.datasets.get_cifar10()

    # setup model
    if args.model == 'MLP3':
        model = MLP3(1000, class_labels)
    elif args.model == 'LeNet':
        model = LeNet(class_labels)
    elif args.model == 'CONV_relu':
        model = CONV_relu(class_labels)
    elif args.model == 'CONV2':
        model = CONV2(class_labels)
    elif args.model == 'CONV3':
        model = CONV3(class_labels)

    model = L.Classifier(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(0.1)
    # optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # trainer
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.ExponentialShift('lr', 0.5), trigger=(25, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
