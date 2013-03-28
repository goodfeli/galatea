from pylearn2.utils import serial
from pylearn2.models.s3c import S3C, Grad_M_Step
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils.call_check import checked_call
from galatea.s3c.s3c import E_Step_Scan
from pylearn2.expr.nnet import inverse_sigmoid_numpy
import numpy as np
from theano import config

dataset = MNIST(which_set = 'train', center = False)

X = dataset.X
y = np.cast['int64'](dataset.y)

means = [ X[y == i,:].mean(axis=0) for i in xrange(y.max()+1) ]


residuals = np.concatenate( [ X[i:i+1,:] - means[y[i]] for i in xrange(X.shape[0]) ], axis=0)
assert len(residuals.shape) == 2

init_beta = 1./(.01+residuals.var(axis=0))
print 'init_beta.shape: ',init_beta.shape

norms = [ np.sqrt(np.square(mu).sum()) for mu in means ]

W = np.zeros( (means[0].shape[0], len(means)), dtype = config.floatX)

for i in xrange(len(means)):
    W[:,i] =  means[i]/norms[i]

init_mu = np.asarray(norms)

model = checked_call(S3C,
        dict(nvis = X.shape[1],
        nhid = len(means),
        init_bias_hid = inverse_sigmoid_numpy(np.asarray([ (y==i).mean() for i in xrange(y.max()+1)])),
        irange = 0.,
        min_B = .1,
        max_B = 1e6,
        min_alpha = .1,
        max_alpha = 1e6,
        m_step = checked_call(Grad_M_Step, dict(learning_rate = 1.)),
        init_mu = init_mu,
        init_alpha = init_mu * 10.,
        init_B = init_beta,
        e_step = E_Step_Scan(
                h_new_coeff_schedule = [ .1 ] * 50,
                s_new_coeff_schedule = [ .1 ] * 50,
                clip_reflections = 1,
                rho = 0.5
            ))
    )

model.W.set_value(W)

model.dataset_yaml_src = '!obj:pylearn2.datasets.mnist.MNIST { which_set : "train", center : 0 }'

serial.save('/u/goodfeli/galatea/pddbm/config/mnist/s3c_hack.pkl',model)
