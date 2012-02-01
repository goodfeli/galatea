#script to demonstrate that theano leaks memory on the gpu

import numpy as np
from pylearn2.models.dbm import DBM
from pylearn2.utils import serial
from pylearn2.models.model import Model
from pylearn2.utils import as_floatX
from pylearn2.utils import sharedX
import warnings
import theano.tensor as T
from theano import config
from theano import function
from theano.gof.op import get_debug_values
from pylearn2.models.s3c import reflection_clip
from pylearn2.models.s3c import damp
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import SufficientStatistics
from galatea.pddbm.pddbm import flatten
import time
import theano
import gc


grads = {}

class PDDBM(Model):

    def __init__(self,
            s3c,
            dbm,
            inference_procedure,
            ):

        super(PDDBM,self).__init__()

        self.s3c = s3c
        s3c.e_step.autonomous = False

        self.dbm = dbm

        self.rng = np.random.RandomState([1,2,3])

        self.s3c.bias_hid = self.dbm.bias_vis

        self.nvis = s3c.nvis

        inference_procedure.register_model(self)
        self.inference_procedure = inference_procedure

        self.num_g = len(self.dbm.W)

        self.dbm.redo_everything()


        for param in self.get_params():
            grads[param] = sharedX(np.zeros(param.get_value().shape))

        self.test_batch_size = 2


        self.s3c.reset_censorship_cache()
        self.s3c.e_step.register_model(self.s3c)

        params_to_approx_grads = self.dbm.get_neg_phase_grads()

        updates = {}

        for param in grads:
            if param in params_to_approx_grads:
                updates[grads[param]] = params_to_approx_grads[param]
            else:
                updates[grads[param]] = T.zeros_like(param)

        sampling_updates = self.dbm.get_sampling_updates()

        for key in sampling_updates:
            assert key not in updates
            updates[key] = sampling_updates[key]

        print 'compiling reset grad func'
        global f
        f = function([], updates = updates)

    def get_params(self):
        return list(set(self.s3c.get_params()).union(set(self.dbm.get_params())))

    def censor_updates(self, updates):

        self.s3c.censor_updates(updates)
        self.dbm.censor_updates(updates)


class PDDBM_InferenceProcedure:

    def __init__(self, schedule,
                       clip_reflections = False,
                       monitor_kl = False,
                       rho = 0.5):
        """Parameters
        --------------
        schedule:
            list of steps. each step can consist of one of the following:
                ['s', <new_coeff>] where <new_coeff> is a number between 0. and 1.
                    does a damped parallel update of s, putting <new_coeff> on the new value of s
                ['h', <new_coeff>] where <new_coeff> is a number between 0. and 1.
                    does a damped parallel update of h, putting <new_coeff> on the new value of h
                ['g', idx]
                    does a block update of g[idx]

        clip_reflections, rho : if clip_reflections is true, the update to Mu1[i,j] is
            bounded on one side by - rho * Mu1[i,j] and unbounded on the other side
        """

        self.schedule = schedule

        self.clip_reflections = clip_reflections
        self.monitor_kl = monitor_kl

        self.rho = as_floatX(rho)

        self.model = None

    def register_model(self, model):
        self.model = model

        self.s3c_e_step = self.model.s3c.e_step

        self.s3c_e_step.clip_reflections = self.clip_reflections
        self.s3c_e_step.rho = self.rho

        self.dbm_ip = self.model.dbm.inference_procedure


    def dbm_observations(self, obs):

        rval = {}
        rval['H_hat'] = obs['G_hat']
        rval['V_hat'] = obs['H_hat']

        return rval

    def truncated_KL(self, V, obs):
        """ KL divergence between variational and true posterior, dropping terms that don't
            depend on the variational parameters """

        s3c_truncated_KL = self.s3c_e_step.truncated_KL(V, obs).mean()

        dbm_obs = self.dbm_observations(obs)

        dbm_truncated_KL = self.dbm_ip.truncated_KL(V = obs['H_hat'], obs = dbm_obs)
        assert len(dbm_truncated_KL.type.broadcastable) == 0

        for s3c_kl_val, dbm_kl_val in get_debug_values(s3c_truncated_KL, dbm_truncated_KL):
            debug_assert( not np.any(np.isnan(s3c_kl_val)))
            debug_assert( not np.any(np.isnan(dbm_kl_val)))

        warnings.warn("""TODO: double check that this decomposition works--
                        It may be ignoring a subtlety where the math for dbm.truncated_kl is based on
                        a fixed V but the pddbm is actually passing it variational parameters
                        for distributional V""")

        rval = s3c_truncated_KL + dbm_truncated_KL

        return rval

    def infer_H_hat(self, V, H_hat, S_hat, G1_hat):
        """
            G1_hat: variational parameters for the g layer closest to h
                    here we use the designation from the math rather than from
                    the list, where it is G_hat[0]
        """

        s3c_presigmoid = self.s3c_e_step.infer_H_hat_presigmoid(V, H_hat, S_hat)

        W = self.model.dbm.W[0]

        top_down = T.dot(G1_hat, W.T)

        presigmoid = s3c_presigmoid + top_down

        H = T.nnet.sigmoid(presigmoid)

        return H

    def infer(self, V, return_history = False):
        """

            return_history: if True:
                                returns a list of dictionaries with
                                showing the history of the variational
                                parameters
                                throughout fixed point updates
                            if False:
                                returns a dictionary containing the final
                                variational parameters
        """

        alpha = self.model.s3c.alpha
        s3c_e_step = self.s3c_e_step
        dbm_ip = self.dbm_ip

        var_s0_hat = 1. / alpha
        var_s1_hat = s3c_e_step.infer_var_s1_hat()

        H_hat = s3c_e_step.init_H_hat(V)
        G_hat = dbm_ip.init_H_hat(H_hat)
        S_hat = s3c_e_step.init_S_hat(V)

        H_hat.name = 'init_H_hat'
        S_hat.name = 'init_S_hat'

        for Hv in get_debug_values(H_hat):

            if Hv.shape[1] != s3c_e_step.model.nhid:
                debug_error_message('H prior has wrong # hu, expected %d actual %d'%\
                        (s3c_e_step.model.nhid,
                            Hv.shape[1]))

        for Sv in get_debug_values(S_hat):
            if isinstance(Sv, tuple):
                warnings.warn("I got a tuple from this and I have no idea why the fuck that happens. Pulling out the single element of the tuple")
                Sv ,= Sv

            if Sv.shape[1] != s3c_e_step.model.nhid:
                debug_error_message('prior has wrong # hu, expected %d actual %d'%\
                        (s3c_e_step.model.nhid,
                            Sv.shape[1]))

        def check_H(my_H, my_V):
            if my_H.dtype != config.floatX:
                raise AssertionError('my_H.dtype should be config.floatX, but they are '
                        ' %s and %s, respectively' % (my_H.dtype, config.floatX))

            allowed_v_types = ['float32']

            if config.floatX == 'float64':
                allowed_v_types.append('float64')

            assert my_V.dtype in allowed_v_types

            if config.compute_test_value != 'off':
                from theano.gof.op import PureOp
                Hv = PureOp._get_test_value(my_H)

                Vv = my_V.tag.test_value

                assert Hv.shape[0] == Vv.shape[0]

        check_H(H_hat,V)

        def make_dict():

            return {
                    'G_hat' : tuple(G_hat),
                    'H_hat' : H_hat,
                    'S_hat' : S_hat,
                    'var_s0_hat' : var_s0_hat,
                    'var_s1_hat': var_s1_hat,
                    }

        history = [ make_dict() ]

        for i, step in enumerate(self.schedule):

            letter, number = step

            coeff = as_floatX(number)

            coeff = T.as_tensor_variable(coeff)

            coeff.name = 'coeff_step_'+str(i)

            if letter == 's':

                new_S_hat = s3c_e_step.infer_S_hat(V, H_hat, S_hat)
                new_S_hat.name = 'new_S_hat_step_'+str(i)

                if self.clip_reflections:
                    clipped_S_hat = reflection_clip(S_hat = S_hat, new_S_hat = new_S_hat, rho = self.rho)
                else:
                    clipped_S_hat = new_S_hat

                S_hat = damp(old = S_hat, new = clipped_S_hat, new_coeff = coeff)

                S_hat.name = 'S_hat_step_'+str(i)

            elif letter == 'h':

                new_H = self.infer_H_hat(V = V, H_hat = H_hat, S_hat = S_hat, G1_hat = G_hat[0])

                new_H.name = 'new_H_step_'+str(i)

                H_hat = damp(old = H_hat, new = new_H, new_coeff = coeff)
                H_hat.name = 'new_H_hat_step_'+str(i)

                check_H(H_hat,V)

            elif letter == 'g':

                if not isinstance(number, int):
                    raise ValueError("Step parameter for 'g' code must be an integer in [0, # g layers) "
                            "but got "+str(number)+" (of type "+str(type(number)))


                G_hat[number] = self.infer_G_hat( H_hat = H_hat, G_hat = G_hat, idx = number)

            else:
                raise ValueError("Invalid inference step code '"+letter+"'. Valid options are 's','h' and 'g'.")

            history.append(make_dict())

        if return_history:
            return history
        else:
            return history[-1]

    def infer_G_hat(self, H_hat, G_hat, idx):
        number = idx
        dbm_ip = self.model.dbm.inference_procedure

        b = self.model.dbm.bias_hid[number]

        W = self.model.dbm.W

        W_below = W[number]

        if number == 0:
            H_hat_below = H_hat
        else:
            H_hat_below = G_hat[number - 1]

        num_g = self.model.num_g

        if number == num_g - 1:
            return dbm_ip.infer_H_hat_one_sided(other_H_hat = H_hat_below, W = W_below, b = b)
        else:
            H_hat_above = G_hat[number + 1]
            W_above = W[number+1]
            return dbm_ip.infer_H_hat_two_sided(H_hat_below = H_hat_below, H_hat_above = H_hat_above,
                                   W_below = W_below, W_above = W_above,
                                   b = b)

model = PDDBM(
        dbm = DBM (
                negative_chains = 100,
                monitor_params = 1,
                rbms = [ serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_2_from_C1_A.pkl") ]
        ),
        s3c =  serial.load("/u/goodfeli/galatea/pddbm/config/stl/full/layer_1_C1.pkl"),
       inference_procedure = PDDBM_InferenceProcedure (
                schedule = [ ['s',1.],   ['h',1.],   ['g',0],   ['h', 0.4], ['s',0.4],
                             ['h',0.4], ['g',0],   ['h',0.4], ['s',0.4],  ['h',0.4],
                             ['g',0],   ['h',0.4], ['s',0.4], ['h', 0.4], ['g',0],
                             ['h',0.4], ['g',0],   ['h',0.4], ['s', 0.4], ['h',0.4] ],
                monitor_kl =  0,
                clip_reflections = 1,
                rho = 0.5
       )
)

before =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
f()
gc.collect(); gc.collect(); gc.collect()
after = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
assert after[0] >= before[0]

