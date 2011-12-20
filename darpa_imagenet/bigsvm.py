# Example of a BaseEstimator implemented with Theano.
# This could use the GPU, except that
# a) linear regression isn't really worth it, and
# b) the multi_hinge_margin Op is only implemented for the CPU.
#
from pylearn2.utils import serial
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
            copy_X=True,
            loss_fn='hinge',
            ):
        # add arguments to class
        self.__dict__.update(locals()); del self.self

    def fit(self, X_train, y_train, X_valid, y_valid):
        batchsize = self.batchsize

        n_train = X_train.shape[0]

        # decrease to a multiple of batchsize
        while n_train % batchsize:
            n_train -= 1

        assert not self.center_and_normalize

        train_features = X_train
        valid_features = X_valid
        train_labels = y_train
        valid_labels = y_valid

        x_i = tensor.matrix(dtype=X_train.dtype)
        y_i = tensor.vector(dtype=y_train.dtype)
        lr = tensor.scalar(dtype=X_train.dtype)

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

            if True:
                n_valid = X_valid.shape[0]
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

#args = which set to use (1 or 2)
#settings dictionary
#save path

which_set = int(sys.argv[1]) -1

assert which_set in [0,1]

f = open(sys.argv[2])
l = f.readlines()
f.close()
s = '\n'.join(l)
args = eval(s)


clf = TheanoSGDClassifier(500, ** args)

def load(featpath,  labelpath):
    print 'loading'
    X = np.load(featpath)
    y = np.load(labelpath)


    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert X.shape[0] == y.shape[0]

    print 'generating mask'
    mask = np.zeros((X.shape[0],),dtype='bool')
    for i in xrange(X.shape[0]):
        mask[i] = ( (i % 1000) < 500 ) == (1-which_set)
        if which_set == 0 and mask[i]:
            if y[i] >= 500:
                print y[i]
                print i
                assert False
            assert y[i] < 500

    print mask.sum(), X.shape
    assert mask.sum() == X.shape[0]/2

    print 'applying mask to X'
    X = X[mask,:]
    print 'applying mask to y'
    y = y[mask]


    print (y.min(), y.max())

    y -= which_set * 500

    assert y.min() == 0
    assert y.max() == 499

    return X,y

print 'getting training data'
X_train, y_train =  load('/mnt/scratch/stitched/train.npy', '/mnt/scratch/stitched/trainl.npy')
print 'getting valid data'
X_valid = np.load( [ '/mnt/scratch/S3C_5625/valid1_S3C_5625.npy', '/mnt/scratch/S3C_5625/valid2_S3C_5625.npy' ] [which_set] )
y_valid = np.load( [ '/mnt/scratch/S3C_5625/validl1_S3C_5625.npy', '/mnt/scratch/S3C_5625/validl2_S3C_5625.npy' ] [which_set] )

y_valid -= which_set * 500

print 'fit'
clf.fit(X_train, y_train, X_valid, y_valid)

print 'save classifier'
serial.save(sys.argv[3], clf)
