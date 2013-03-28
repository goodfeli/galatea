#python bigsvm.py config.txt

# Example of a BaseEstimator implemented with Theano.
# This could use the GPU, except that
# a) linear regression isn't really worth it, and
# b) the multi_hinge_margin Op is only implemented for the CPU.
#
import numpy as np
from theano import config
from scipy import io
from pylearn2.utils import sharedX
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
from theano.tensor.shared_randomstreams import RandomStreams

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
            prob_max_pool=True,
            n_pools = 2000
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

        if self.prob_max_pool:
            theano_rng = RandomStreams(85)
            I = sharedX( np.ones( (1,train_features.shape[1], self.n_pools)))
            assert batchsize == 1
            M = theano_rng.binomial( size = I.shape, n = 1, p = I, dtype = I.dtype)
            prod = x_i.dimshuffle((0,1,'x')) * M
            z_i = prod.max(axis=1)
            assert z_i.ndim == 2
            logreg_n_in = self.n_pools
        else:
            z_i = x_i
            logreg_n_in = train_features.shape[1]


        y_i = tensor.vector(dtype=y.dtype)
        lr = tensor.scalar(dtype=X.dtype)

        feature_logreg = LogisticRegression.new(z_i,
                n_in = logreg_n_in, n_out=self.n_classes,
                dtype=z_i.dtype)

        if self.loss_fn=='log':
            traincost = feature_logreg.nll(y_i).sum()
        elif self.loss_fn=='hinge':
            raw_output = tensor.dot(feature_logreg.input, feature_logreg.w)+feature_logreg.b
            traincost = multi_hinge_margin(raw_output, y_i).sum()
        else:
            raise NotImplementedError(self.loss_fn)
        traincost = traincost + abs(feature_logreg.w).sum() * self.l1_regularization
        traincost = traincost + (feature_logreg.w**2).sum() * self.l2_regularization

        params = [ elem for elem in feature_logreg.params ]
        grads = tensor.grad(traincost, params)

        updates = updates=pylearn.gd.sgd.sgd_updates(
                    params=feature_logreg.params,
                    grads=tensor.grad(traincost, feature_logreg.params),
                    stepsizes=[lr/batchsize,lr/(10*batchsize)])


        if self.prob_max_pool:
            zero = np.cast[config.floatX](0.0)
            one = np.cast[config.floatX](1.0)
            two = np.cast[config.floatX](2.0)
            #approximation based on one sample
            assert two.dtype == I.dtype
            assert M.dtype == I.dtype
            assert one.dtype == I.dtype
            approx_grad = traincost * (two * M - one)/ ( I*M + (one-I)*(one-M))
            #traincost doesn't respect dtype
            approx_grad = tensor.cast(approx_grad, config.floatX)
            step_size = tensor.cast(lr,config.floatX)/np.cast[config.floatX](batchsize)
            assert step_size.dtype == I.dtype
            assert approx_grad.dtype == I.dtype
            update = tensor.clip(I - step_size*approx_grad,zero,one)
            updates.append( (I,update) )


        train_logreg_fn = theano.function([x_i, y_i, lr],
                [feature_logreg.nll(y_i).mean(),
                    feature_logreg.errors(y_i).mean()],
                updates = updates
                )

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

f = open(sys.argv[1])
l = f.readlines()
f.close()
s = '\n'.join(l)
args = eval(s)


clf = TheanoSGDClassifier(10, ** args)

dbm_feat = io.loadmat('../pddbm/dbm_feat.mat')
X = dbm_feat['train_X']
H = dbm_feat['train_H']
G = dbm_feat['train_G']
Xt = dbm_feat['test_X']
Ht = dbm_feat['test_H']
Gt = dbm_feat['test_G']





from pylearn2.datasets.mnist import MNIST
X = np.concatenate((X,H,G),axis=1)
y = MNIST(which_set="train").y.astype(int)
print 'fit'
clf.fit(X, y)

del X
del H
del G
del y

y = np.asarray(MNIST(which_set="test").y).astype(int)
X = np.concatenate((Xt,Ht,Gt),axis=1)
print 'evaluating svm'
yhat = clf.predict(X)

print (yhat == y).mean()
