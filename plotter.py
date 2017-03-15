import matplotlib.pyplot as plt
import json

def plot_result(logfiles, target='validation/main/accuracy', outfile=None):
    if not target in ['main/loss', 'main/accuracy',
                      'validation/main/loss', 'validation/main/accuracy']:
        print('invalid target: {}'.format(target))
        exit(1)

    fig, ax = plt.subplots()

    for label, logfile in logfiles:
        result = json.load(open(logfile))
        epoch = []
        loss = []
        for x in result:
            epoch.append(x['epoch'])
            loss.append(x[target])
        ax.plot(epoch, loss, label=label, marker='.')

    ax.set_xlabel('epoch')
    ax.set_ylabel(target)
    ax.legend(loc='best')
    ax.grid(True)
    fig.tight_layout()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()

if __name__ == '__main__':
    logfiles = [('MLP3', 'result_MLP3/log'),
                ('LeNet (mSGD)', 'result_LeNet_mSGD/log'),
                ('LeNet (Adam)', 'result_LeNet_Adam/log'),
                ('CONV_relu', 'result_CONV_relu/log'),
                ('CONV2', 'result_CONV2/log')]

    plot_result(logfiles, 'main/loss', 'cifar10_loss.png')
    plot_result(logfiles, 'main/accuracy', 'cifar10_acc.png')
    plot_result(logfiles, 'validation/main/loss', 'cifar10_val_loss.png')
    plot_result(logfiles, 'validation/main/accuracy', 'cifar10_val_acc.png')
