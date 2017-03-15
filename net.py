import chainer
import chainer.functions as F
import chainer.links as L


class MLP3(chainer.Chain):

    def __init__(self, n_units, n_out=10, activation=F.relu):
        self.activation = activation
        super(MLP3, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_out),
        )

    def __call__(self, x):
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        return self.l3(h2)


class LeNet(chainer.Chain):
    # http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

    def __init__(self, class_labels):
        super(LeNet, self).__init__(
            conv1=F.Convolution2D(None, 6, ksize=5),
            conv2=F.Convolution2D(None, 16, ksize=5),
            fc1=F.Linear(None, 120),
            fc2=F.Linear(None, 84),
            fc3=F.Linear(None, class_labels)
        )
        self.train = True

    def __call__(self, x):
        h = self.conv1(x)
        # print("conv1", h.shape)
        h = F.max_pooling_2d(h, ksize=2)
        h = F.sigmoid(h)
        # print("pool1", h.shape)

        h = self.conv2(h)
        # print("conv2", h.shape)
        h = F.max_pooling_2d(h, ksize=2)
        # print("pool2", h.shape)
        h = F.sigmoid(h)
        h = F.dropout(h, ratio=0.5, train=self.train)

        h = self.fc1(h)
        # print("fc1", h.shape)

        h = self.fc2(h)
        # print("fc2", h.shape)

        y = self.fc3(h)
        # print("y", y.shape)

        return y
