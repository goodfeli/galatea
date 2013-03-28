# Example of a BaseEstimator implemented with Theano.
# This could use the GPU, except that
# a) linear regression isn't really worth it, and
# b) the multi_hinge_margin Op is only implemented for the CPU.
#
import numpy as np
try:
    import theano
    from theano import tensor
except ImportError:
    print('Failed to import Theano - see installation instructions at '
            'http://www.deeplearning.net/software/theano/')
    raise
try:
    from pylearn.shared.layers.logreg import LogisticRegression
    import pylearn.gd.sgd
    from pylearn.formulas.costs import multi_hinge_margin
except ImportError:
    print('Failed to import pylearn - clone it from https://hg.assembla.com/pylearn')
    raise

class TheanoSGDClassifier(object):
    def __init__(self,
            n_classes,
            batchsize=100,
            learnrate = 0.005,
            l1_regularization = 0.0,
            l2_regularization = 0.0,
            min_feature_std =0.3,
            n_epochs = 100,
            anneal_epoch=20,
            center_and_normalize=False,
            validset_fraction=.2,
            validset_max_examples=5000,
            copy_X=True,
            loss_fn='hinge',
            ):
        # add arguments to class
        self.__dict__.update(locals()); del self.self

    def fit(self, X, y):
        batchsize = self.batchsize

        n_valid = int(min(self.validset_max_examples, self.validset_fraction * X.shape[0]))
        # increase to a multiple of batchsize
        while n_valid % batchsize:
            n_valid += 1

        n_train = X.shape[0] - n_valid

        # decrease to a multiple of batchsize
        while n_train % batchsize:
            n_train -= 1

        if self.center_and_normalize and self.copy_X:
            X = X.copy()

        train_features = X[:n_train]
        valid_features = X[n_train:]
        train_labels = y[:n_train]
        valid_labels = y[n_train:]

        if self.center_and_normalize:
            print("Computing mean and std.dev")

            #this loop seems more memory efficient than numpy
            m= np.zeros(train_features.shape[1])
            msq= np.zeros(train_features.shape[1])
            for i in xrange(train_features.shape[0]):
                alpha = 1.0 / (i+1)
                v = train_features[i]
                m = alpha * v + (1-alpha)*m
                msq = alpha * v*v + (1-alpha)*msq

            self.X_mean_ = theano.shared(m.astype(X.dtype))
            self.X_std_ = theano.shared(
                    np.maximum(
                        self.min_feature_std,
                        np.sqrt(msq - m*m)).astype(X.dtype))

            X -= self.X_mean_.get_value()
            X /= self.X_std_.get_value()

        x_i = tensor.matrix(dtype=X.dtype)
        y_i = tensor.vector(dtype=y.dtype)
        lr = tensor.scalar(dtype=X.dtype)

        feature_logreg = LogisticRegression.new(x_i,
                n_in = train_features.shape[1], n_out=self.n_classes,
                dtype=x_i.dtype)

        if self.loss_fn=='log':
            traincost = feature_logreg.nll(y_i).sum()
        elif self.loss_fn=='hinge':
            raw_output = tensor.dot(feature_logreg.input, feature_logreg.w)+feature_logreg.b
            traincost = multi_hinge_margin(raw_output, y_i).sum()
        else:
            raise NotImplementedError(self.loss_fn)
        traincost = traincost + abs(feature_logreg.w).sum() * self.l1_regularization
        traincost = traincost + (feature_logreg.w**2).sum() * self.l2_regularization
        train_logreg_fn = theano.function([x_i, y_i, lr],
                [feature_logreg.nll(y_i).mean(),
                    feature_logreg.errors(y_i).mean()],
                updates=pylearn.gd.sgd.sgd_updates(
                    params=feature_logreg.params,
                    grads=tensor.grad(traincost, feature_logreg.params),
                    stepsizes=[lr/batchsize,lr/(10*batchsize)]))

        test_logreg_fn = theano.function([x_i, y_i],
                feature_logreg.errors(y_i))

        if self.center_and_normalize:
            feature_logreg_test = LogisticRegression(
                    (x_i - self.X_mean_)/self.X_std_,
                    feature_logreg.w,
                    feature_logreg.b)
            self.predict_fn_ = theano.function([x_i], feature_logreg_test.argmax)
        else:
            self.predict_fn_ = theano.function([x_i], feature_logreg.argmax)

        best_epoch = -1
        best_epoch_valid = -1
        best_epoch_train = -1
        best_epoch_test = -1
        valid_rate=-1
        test_rate=-1
        train_rate=-1

        for epoch in xrange(self.n_epochs):
            # validate
            # Marc'Aurelio, you crazy!!
            # the division by batchsize is done in the cost function
            e_lr = np.float32(self.learnrate / max(1.0, np.floor(max(1.,
                (epoch+1)/float(self.anneal_epoch))-2)))

            if n_valid:
                l01s = []
                for i in xrange(n_valid/batchsize):
                    x_i = valid_features[i*batchsize:(i+1)*batchsize]
                    y_i = valid_labels[i*batchsize:(i+1)*batchsize]

                    #lr=0.0 -> no learning, safe for validation set
                    l01 = test_logreg_fn((x_i), y_i)
                    l01s.append(l01)
                valid_rate = 1-np.mean(l01s)
                #print('Epoch %i validation accuracy: %f'%(epoch, valid_rate))

                if valid_rate > best_epoch_valid:
                    best_epoch = epoch
                    best_epoch_test = test_rate
                    best_epoch_valid = valid_rate
                    best_epoch_train = train_rate

                print('Epoch=%i best epoch %i valid %f test %f best train %f current train %f'%(
                    epoch, best_epoch, best_epoch_valid, best_epoch_test, best_epoch_train, train_rate))
                if epoch > self.anneal_epoch and epoch > 2*best_epoch:
                    break
            else:
                print('Epoch=%i current train %f'%( epoch, train_rate))

            #train
            l01s = []
            nlls = []
            for i in xrange(n_train/batchsize):
                x_i = train_features[i*batchsize:(i+1)*batchsize]
                y_i = train_labels[i*batchsize:(i+1)*batchsize]
                nll, l01 = train_logreg_fn((x_i), y_i, e_lr)
                nlls.append(nll)
                l01s.append(l01)
            train_rate = 1-np.mean(l01s)
            #print('Epoch %i train accuracy: %f'%(epoch, train_rate))

    def predict(self, X):
        return self.predict_fn_(X)

import sys

f = open(sys.argv[2])
l = f.readlines()
f.close()
s = '\n'.join(l)
args = eval(s)


clf = TheanoSGDClassifier(100, ** args)
from pylearn2.datasets.cifar100 import CIFAR100
X = np.load(sys.argv[1])
if len(X.shape) == 4:
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])
y = CIFAR100(which_set="train").y_fine.astype(int)
print 'fit'
clf.fit(X, y)

del X
del y

y = np.asarray(CIFAR100(which_set="test").y_fine).astype(int)
print 'loading test data'
X = np.load(sys.argv[3])
X = X.reshape(X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])

print 'evaluating svm'
yhat = clf.predict(X)

print (yhat == y).mean()
