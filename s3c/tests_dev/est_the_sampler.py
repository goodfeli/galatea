import os
from galatea.s3c.s3c import S3C
from galatea.s3c.s3c import SufficientStatistics
from galatea.s3c.s3c import SufficientStatisticsHolder
from galatea.s3c.s3c import VHS_E_Step
from galatea.s3c.s3c import VHS_Solve_M_Step
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.utils import as_floatX
from theano import function
import numpy as np
import theano.tensor as T
import copy
from theano.printing import Print
from theano import config
from pylearn2.utils import serial
from theano.sandbox.linalg.ops import alloc_diag, extract_diag, matrix_inverse
config.compute_test_value = 'raise'
from theano.tensor.shared_randomstreams import RandomStreams

def broadcast(mat, shape_0):
    rval = mat
    if mat.shape[0] != shape_0:
        assert mat.shape[0] == 1

        rval = np.zeros((shape_0, mat.shape[1]),dtype=mat.dtype)

        for i in xrange(shape_0):
            rval[i,:] = mat[0,:]

    return rval


class TestWithFiniteSamples:
    def __init__(self):
        """ gets a small batch of data
            sets up an S3C model and learns on the data
            creates an expression for the log likelihood of the data
        """

        self.tol = 1e-5

        goodfeli_tmp = os.environ['GOODFELI_TMP']
        dataset = serial.load(goodfeli_tmp + '/cifar10_preprocessed_train_2M.pkl')

        X = dataset.get_batch_design(1)
        X = X[:,0:5]
        X -= X.mean()
        X /= X.std()
        m, D = X.shape
        N = 2

        self.model = S3C(nvis = D,
                         nhid = N,
                         irange = 1.,
                         init_bias_hid = -.1,
                         init_B = 3.,
                         min_B = 1e-8,
                         max_B = 1000.,
                         init_alpha = 1., min_alpha = 1e-8, max_alpha = 1000.,
                         init_mu = 1., e_step = VHS_E_Step(h_new_coeff_schedule = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1. ]),
                         new_stat_coeff = 1.,
                         m_step = VHS_Solve_M_Step( new_coeff = 1.0 ),
                         W_eps = 0., mu_eps = 1e-8,
                         min_bias_hid = -1e30, max_bias_hid = 1e30,
                        learn_after = None)


        #print "ORTHOGONAL WEIGHTS HACK"
        #self.model.W.set_value( np.cast[self.model.W.dtype] ( np.identity(max(N,D))[0:D,0:N] ))


        self.X = X
        self.N = N
        self.m = m


    def test_expected_energy_vhs(self):

        "tests that the analytical expression for the expected energy of v,h, and s matches a finite sample approximation to it "

        assert self.m == 1

        model = self.model
        e_step = model.e_step
        V = self.X
        rng = np.random.RandomState([1.,2.,3.])

        H = np.cast[config.floatX](rng.uniform(0.0,1.0,(self.m,self.N)))
        Mu1 = np.cast[config.floatX](rng.uniform(-5.0,5.0,(self.m,self.N)))

        np.save('H_sampler.npy',H)
        np.save('Mu1_sampler.npy',Mu1)

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H

        Mu1_var = T.matrix(name='Mu1')
        Mu1_var.tag.test_value = Mu1

        sigma0 = 1. / model.alpha
        Sigma1 = e_step.mean_field_Sigma1()

        mu0 = T.zeros_like(self.model.mu)


        theano_rng = RandomStreams(rng.randint(2**30))

        H_sample = theano_rng.binomial( size = H_var.shape, n = 1, p =  H_var)

        pos_sample = theano_rng.normal( size = H_var.shape, avg = Mu1, std = T.sqrt(Sigma1) )
        neg_sample = theano_rng.normal( size = H_var.shape, avg = 0.0, std = T.sqrt(1./model.alpha) )


        thousand = 1000
        million = thousand * thousand

        num_samples = 40 * million

        H_samples = np.zeros((num_samples,self.N))
        pos_samples = np.zeros((num_samples,self.N))
        neg_samples = np.zeros((num_samples,self.N))

        sample_func = function([H_var, Mu1_var],[H_sample, pos_sample, neg_sample] )

        for i in xrange(num_samples):
            if i % 10000 == 0:
                print 'sample ',i

            H_samples[i:i+1,:], pos_samples[i:i+1,:], neg_samples[i:i+1,:] = sample_func(H, Mu1)

        np.save('H_samples.npy', H_samples)
        np.save('pos_samples.npy',pos_samples)
        np.save('neg_samples.npy',neg_samples)



if __name__ == '__main__':
    obj = TestWithFiniteSamples()

    obj.test_expected_energy_vhs()
