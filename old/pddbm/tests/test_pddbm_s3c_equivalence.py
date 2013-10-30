from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import E_Step
from pylearn2.models.s3c import Grad_M_Step
from galatea.pddbm.pddbm import PDDBM, InferenceProcedure
from pylearn2.expr.information_theory import entropy_binary_vector
from pylearn2.models.rbm import RBM
from pylearn2.models.dbm import DBM
from theano import function
import numpy as np
import theano.tensor as T
from theano import config
#from pylearn2.utils import serial
import warnings

if config.floatX != 'float64':
    config.floatX = 'float64'
    warnings.warn("Changed config.floatX to float64. s3c inference tests currently fail due to numerical issues for float32")

def broadcast(mat, shape_0):

    assert len(mat.shape) == 1

    rval = np.zeros((shape_0, mat.shape[0]),dtype=mat.dtype) + mat

    return rval


def make_e_step_from_inference_procedure(ip):

    #we used to
    #transcribe the pd-dbm inference schedule
    #because dbm weights are fixed to 0, inference on g
    #has no effect and we can just omit it
    #we can only allow alternating s and h updates starting with s
    #because that's how the S3C E step is implemented

    """
    h_new_coeff_schedule = []
    s_new_coeff_schedule = []

    seeking = 's'

    for elem in ip.schedule:

        if seeking == 's':
            assert elem[0] in ['g','s']
            if elem[0] == 's':
                s_new_coeff_schedule.append(elem[1])
                seeking = 'h'
        elif seeking == 'h':
            assert elem[0] in ['g','h']
            if elem[0] == 'h':
                h_new_coeff_schedule.append(elem[1])
                seeking = 's'

    """

    #now the pddbm no longer uses a fixed schedule but s3c
    #still does so we just give s3c an arbitrary schedule
    #(the one used for the PDDBM in the old version of the
    #test)

    h_new_coeff_schedule = [ .1, .2, .3, .4, .4, .5, .5 ]
    s_new_coeff_schedule = [ .1, .2, .3, .4, .4, .5, .1 ]

    clip_reflections = ip.clip_reflections
    rho = ip.rho

    return E_Step(
            h_new_coeff_schedule = h_new_coeff_schedule,
            s_new_coeff_schedule = s_new_coeff_schedule,
            clip_reflections = clip_reflections,
            rho = rho)


def sigmoid(x):
    return 1./(1.+np.exp(-x))

class Test_PDDBM_S3C_Equivalence:
    def __init__(self):
        """ gets a small batch of data
            sets up a PD-DBM model with its DBM weights set to 0
            so that it represents the same distribution as an S3C
            model
            Makes the S3C model it matches
            (Note that their learning rules don't match since the
            complete partition function of the S3C model is tractable
            but the PD-DBM has to approximate the h partition function
            via sampling)
        """

        self.tol = 1e-5

        X = np.random.RandomState([1,2,3]).randn(1000,5)

        X -= X.mean()
        X /= X.std()
        m, D = X.shape
        N = 6
        N2 = 7

        self.X = X
        self.N = N
        self.N2 = N2
        self.m = m
        self.D = D


        s3c_for_pddbm = self.make_s3c()
        self.s3c = self.make_s3c()

        self.s3c.W.set_value(s3c_for_pddbm.W.get_value())

        rbm = RBM(nvis = N, nhid = N2, irange = .0, init_bias_vis = -1.5, init_bias_hid = 6.)

        #don't give it an inference procedure so it won't make a learn_func
        self.model = PDDBM(
                dbm = DBM(  use_cd = 1,
                            rbms = [ rbm  ]),
                s3c = s3c_for_pddbm
        )

        self.inference_procedure = InferenceProcedure( clip_reflections = True, rho = 0.5 )

        self.model.make_pseudoparams()

        self.inference_procedure.register_model(self.model)
        self.inference_procedure.redo_theano()

        self.e_step = make_e_step_from_inference_procedure(self.inference_procedure)

        self.e_step.register_model(self.s3c)

        self.s3c.make_pseudoparams()

        #check that all the parameters match
        assert np.abs(self.s3c.W.get_value() - self.model.s3c.W.get_value()).max() == 0.0
        assert np.abs(self.s3c.bias_hid.get_value() - self.model.s3c.bias_hid.get_value()).max() == 0.0
        assert np.abs(self.s3c.alpha.get_value() - self.model.s3c.alpha.get_value()).max() == 0.0
        assert np.abs(self.s3c.mu.get_value() - self.model.s3c.mu.get_value()).max() == 0.0
        assert np.abs(self.s3c.B_driver.get_value() - self.model.s3c.B_driver.get_value()).max() == 0.0

        #check that the assumptions making these tests valid are met
        assert np.abs(self.model.dbm.W[0].get_value()).max() == 0.0

    def make_s3c(self):

        return S3C(nvis = self.D,
                 nhid = self.N,
                 irange = .1,
                 init_bias_hid = -1.5,
                 init_B = 3.,
                 min_B = 1e-8,
                 max_B = 1000.,
                 init_alpha = 1., min_alpha = 1e-8, max_alpha = 1000.,
                 init_mu = 1., e_step = None,
                 m_step = Grad_M_Step(),
                 min_bias_hid = -1e30, max_bias_hid = 1e30,
                )


    def test_kl_equivalence(self):

        "tests that the kl divergence for the two models is the same "

        """ This is a tricky task. The full KL-divergence is not tractable,
        but this is the quantity that's known to be the same for the two models
        (since the PDDBM should have 0 KL-divergence from g, since its weights
        are fixed to 0). The quantity we monitor inside the models is the
        "truncated KL divergence", the portion that depends on the variational
        parameters. In this case (S3C / PD-DBM with DBM weights fixed to 0) the
        partition function is also tractable, so we can include the terms that
        depend on the partition function. Fortunately this is enough of the
        KL divergence to guarantee that the quantity is the same for both models.
        There's another term that depends on P(v) which is still intractable but
        g has no effect on P(v) in this case since the DBM weights are fixed to
        0. """


        """

        Let Z represent all latent vars, V all visible vars

        KL(Q(Z)||P(Z|v)) = \sum_z Q(z) log Q(z) / P(z | v)
                         = \sum_z Q(z) log Q(z) - \sum_z Q(z) log P(z | v)
                         = - H_Q(Z) - \sum_z Q(z) log P(z,v) + sum_z Q(z) log P(v)
                         = - H_Q(Z) - \sum_z Q(z) log exp(-E(z,v))/Z + log P(v)
                         = - H_Q(Z) - \sum_z Q(z) log exp(-E(z,v))
                                    + \sum_z Q(z) Z + log P(v)
                         = - H_Q(Z) + \sum_z Q(z) E(z,v) + log Z + log P(v)
                         = - H_Q(Z) + E_{z\simQ}[E(z,v)] + log Z + log P(v)

        """


        model = self.model
        ip = self.inference_procedure
        e_step = self.e_step
        X = self.X

        assert X.shape[0] == self.m

        H = np.cast[config.floatX](self.model.rng.uniform(0.,1.,(self.m, self.N)))
        S = np.cast[config.floatX](self.model.rng.uniform(-5.,5.,(self.m, self.N)))
        G = np.cast[config.floatX](
                broadcast(
                    sigmoid(self.model.dbm.rbms[0].bias_hid.get_value()), self.m))

        H_var = T.matrix(name='H_var')
        H_var.tag.test_value = H
        S_var = T.matrix(name='S_var')
        S_var.tag.test_value = S
        G_var = T.matrix(name='G_var')
        G_var.tag.test_value = G


        dbm_sigma0 = ip.infer_var_s0_hat()
        dbm_Sigma1 = ip.infer_var_s1_hat()

        dbm_trunc_kl = ip.truncated_KL( V = X, obs = { 'H_hat' : H_var,
                                                 'S_hat' : S_var,
                                                 'var_s0_hat' : dbm_sigma0,
                                                 'var_s1_hat' : dbm_Sigma1,
                                                 'G_hat' : ( G_var, ) } ).mean()

        #just the part related to G (check that it all comes out to 0)
        #dbm_trunc_kl = - entropy_binary_vector( G_var ).mean() - T.dot(G_var.mean(axis=0),self.model.dbm.rbms[0].bias_hid)

        assert len(dbm_trunc_kl.type.broadcastable) == 0


        s3c_sigma0 = e_step.infer_var_s0_hat()
        s3c_Sigma1 = e_step.infer_var_s1_hat()
        s3c_mu0 = T.zeros_like(self.s3c.mu)

        s3c_trunc_kl = e_step.truncated_KL( V = X, obs = { 'H_hat' : H_var,
            'S_hat' : S_var,
            'var_s0_hat' : s3c_sigma0,
            'var_s1_hat' : s3c_Sigma1 } )


        dbm_log_partition_function = self.model.s3c.log_partition_function() \
                + T.nnet.softplus(self.model.dbm.rbms[0].bias_hid).sum()

        #just the part related to G (check that it all comes out to 0)
        #dbm_log_partition_function = T.nnet.softplus(self.model.dbm.rbms[0].bias_hid).sum()

        s3c_log_partition_function = self.s3c.log_partition_function()

        s3c_partial_kl = s3c_trunc_kl.mean() + s3c_log_partition_function
        assert len(s3c_partial_kl.type.broadcastable) == 0
        dbm_partial_kl = dbm_trunc_kl + dbm_log_partition_function

        s3c_partial_kl, dbm_partial_kl = function([H_var,S_var,G_var],
                (s3c_partial_kl, dbm_partial_kl))(H,S,G)


        print s3c_partial_kl
        print dbm_partial_kl

        assert np.allclose(s3c_partial_kl, dbm_partial_kl)


    def test_inference_equivalence(self):

        s3c_obs = self.e_step.variational_inference(self.X)
        pddbm_obs = self.inference_procedure.hidden_obs

        self.inference_procedure.update_var_params(self.X)

        s3c_H, s3c_S, pddbm_H, pddbm_S = function([],[s3c_obs['H_hat'],
            s3c_obs['S_hat'], pddbm_obs['H_hat'], pddbm_obs['S_hat']])()

        raise AssertionError("""Known failure: PDDBM and S3C don't currently have
                equivalent inference implementations so their inference procedures
                get different results""")

        assert np.allclose(s3c_H,pddbm_H)
        assert np.allclose(s3c_S, pddbm_S)


