import warnings
import os
from galatea.s3c.s3c import S3C
from galatea.s3c.s3c import VHS_E_Step
from galatea.s3c.s3c import VHS_Solve_M_Step
from theano import function
import numpy as np
import theano.tensor as T
from theano.printing import Print
from theano import config
from pylearn2.utils import serial
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

        X = dataset.get_batch_design(1000)
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

    def test_entropy_hs(self):

        "tests that the analytical expression for the entropy matches a finite sample approximation to it "

        model = self.model
        e_step = model.e_step
        rng = np.random.RandomState([1.,2.,3.])

        H = np.cast[config.floatX](rng.uniform(0.0,1.0,(self.m,self.N)))

        Mu1 = np.cast[config.floatX](rng.uniform(-5.0,5.0,(self.m,self.N)))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H

        Mu1_var = T.matrix(name='Mu1')
        Mu1_var.tag.test_value = Mu1

        sigma0 = 1. / model.alpha
        Sigma1 = e_step.mean_field_Sigma1()

        analytical_entropy = model.entropy_hs(H = H_var, sigma0 = sigma0, Sigma1 = Sigma1)


        theano_rng = RandomStreams(rng.randint(2**30))

        H_sample = theano_rng.binomial( size = H_var.shape, n = 1, p =  H_var)

        pos_sample = theano_rng.normal( size = H_var.shape, avg = Mu1, std = T.sqrt(Sigma1) )
        neg_sample = theano_rng.normal( size = H_var.shape, avg = 0.0, std = T.sqrt(1./model.alpha) )

        final_sample = H_sample * pos_sample + (1.-H_sample)*neg_sample

        """
            log P(h,s)
            = log P(h) + log P(s|h)
            = log (p*h + (1-p)*(1-h) ) + h log sqrt( 1 / ( sigma two pi ) ) exp( (x - mu1)^2 / (2 sigma) ) + (1-h) log sqrt ( alpha / two pi) exp( alpha x^2 / 2)
            = log (2 p*h + 1-h -p  ) + h log sqrt( 1 / ( sigma two pi ) ) exp( (x - mu1)^2 / (2 sigma) ) + (1-h) log sqrt ( alpha / two pi) exp( alpha x^2 / 2)
            = log (2 p*h + 1-h -p  ) + h [  (x - mu1)^2 / (2 sigma) - 0.5 log (sigma two pi) ] + (1-h) [ 0.5 log alpha + alpha x^2 / two - 0.5 log (two pi) ]
        """

        log_prob_H = T.log(H_var * H_sample + (1.-H_var)*(1.-H_sample))

        log_prob_S_given_H1 = - T.sqr( final_sample - Mu1) / (2. * Sigma1) - 0.5 * T.log( Sigma1 * 2. * np.pi)
        log_prob_S_given_H0 = - T.sqr( final_sample ) * model.alpha / 2. + 0.5 * T.log(model.alpha) - 0.5 * T.log ( 2. * np.pi)
        log_prob_S = H_sample * log_prob_S_given_H1 + (1.-H_sample)*log_prob_S_given_H0

        log_prob = (log_prob_H + log_prob_S).sum(axis=1)

        log_prob_func = function([H_var,Mu1_var],log_prob)

        analytical_entropy = function([H_var,Mu1_var],analytical_entropy)(H,Mu1)

        print 'analytical entropy ',analytical_entropy.mean()

        num_samples = 3000
        approx_entropy_prediv = np.zeros(analytical_entropy.shape, dtype=analytical_entropy.dtype)

        for i in xrange(num_samples):
            warnings.warn('not numerically stable, see the energy test for how to redo this')
            approx_entropy_prediv -= log_prob_func(H, Mu1)

            approx_entropy = approx_entropy_prediv / float(i+1)

            print 'ave diff: ',np.abs(approx_entropy - analytical_entropy).mean()
            print 'diff std: ',np.abs(approx_entropy - analytical_entropy).std()

        M = np.zeros((self.m,2))
        M[:,0] = analytical_entropy
        M[:,1] = approx_entropy

        print M



    def test_entropy_h(self):

        "tests that the analytical expression for the entropy matches a finite sample approximation to it "

        model = self.model


        rng = np.random.RandomState([1.,2.,3.])

        H = np.cast[config.floatX](rng.uniform(0.0,1.0,(self.m,self.N)))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H

        analytical_entropy = model.entropy_h(H = H_var)


        theano_rng = RandomStreams(rng.randint(2**30))

        H_sample = theano_rng.binomial( size = H_var.shape, n = 1, p =  H_var)

        log_prob_H = T.log(H_var * H_sample + (1.-H_var)*(1.-H_sample))

        log_prob = log_prob_H.sum(axis=1)

        log_prob_func = function([H_var],log_prob)

        analytical_entropy = function([H_var],analytical_entropy)(H)

        num_samples = 3000
        approx_entropy_prediv = np.zeros(analytical_entropy.shape, dtype=analytical_entropy.dtype)

        for i in xrange(num_samples):
            warnings.warn('not numerically stable, see the energy test for how to redo this')
            approx_entropy_prediv -= log_prob_func(H)

            approx_entropy = approx_entropy_prediv / float(i+1)

            print 'ave diff: ',np.abs(approx_entropy - analytical_entropy).mean()

        M = np.zeros((self.m,2))
        M[:,0] = analytical_entropy
        M[:,1] = approx_entropy

        print M

    def test_expected_energy_vhs(self):

        "tests that the analytical expression for the expected energy of v,h, and s matches a finite sample approximation to it "

        model = self.model
        e_step = model.e_step
        V = self.X
        rng = np.random.RandomState([1.,2.,3.])

        H = np.cast[config.floatX](rng.uniform(0.0,1.0,(self.m,self.N)))

        Mu1 = np.cast[config.floatX](rng.uniform(-5.0,5.0,(self.m,self.N)))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H

        Mu1_var = T.matrix(name='Mu1')
        Mu1_var.tag.test_value = Mu1

        sigma0 = 1. / model.alpha
        Sigma1 = e_step.mean_field_Sigma1()

        mu0 = T.zeros_like(self.model.mu)

        analytical_energy = model.expected_energy_vhs(V, H_var, mu0, Mu1_var, sigma0, Sigma1)

        theano_rng = RandomStreams(rng.randint(2**30))

        H_sample = theano_rng.binomial( size = H_var.shape, n = 1, p =  H_var)

        pos_sample = theano_rng.normal( size = H_var.shape, avg = Mu1, std = T.sqrt(Sigma1) )
        neg_sample = theano_rng.normal( size = H_var.shape, avg = 0.0, std = T.sqrt(1./model.alpha) )

        final_sample = H_sample * pos_sample + (1.-H_sample)*neg_sample

        energy = model.energy_vhs(V, H_sample, final_sample)

        energy_func = function([H_var,Mu1_var],energy)

        analytical_energy = function([H_var,Mu1_var],analytical_energy)(H,Mu1)

        print 'analytical entropy ',analytical_energy.mean()

        two = 2
        thousand = 1000
        million = thousand * thousand

        num_samples = 2 * thousand * thousand

        approx_energy = np.zeros(analytical_energy.shape, dtype=analytical_energy.dtype)

        write_freq = 100

        diffs = np.zeros(num_samples/write_freq)
        x = np.zeros(num_samples/write_freq)

        record = np.zeros(num_samples/write_freq)

        for i in xrange(num_samples):
            energy_sample = energy_func(H, Mu1)
            approx_energy += (energy_sample - approx_energy) / float(i+1)

            if i % write_freq == 0:
                x[i/write_freq] = i
                diffs[i/write_freq] = np.abs(approx_energy - analytical_energy).mean()
                record[i/write_freq] = approx_energy[0]


            if i % 10000 == 0:
                print 'ave diff: ',np.abs(approx_energy - analytical_energy).mean()
                print 'diff std: ',np.abs(approx_energy - analytical_energy).std()

        analytical = np.zeros(tuple()) + analytical_energy[0]
        np.save('H.npy',H)
        np.save('Mu1.npy',Mu1)
        np.save('analytical.npy',analytical)
        np.save('record.npy', record)
        np.save('x.npy', x)
        np.save('diffs.npy',diffs)
        np.save('final_diffs.npy', np.abs(approx_energy - analytical_energy) )

        import matplotlib.pyplot as plt
        plt.plot(x,diffs)
        plt.show()

        M = np.zeros((self.m,2))
        M[:,0] = analytical_energy
        M[:,1] = approx_energy

        print M

if __name__ == '__main__':
    obj = TestWithFiniteSamples()

    obj.test_expected_energy_vhs()
