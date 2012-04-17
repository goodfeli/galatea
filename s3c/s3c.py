__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2011, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

#Included so that old imports using this filename
#still work. All development on these classes should
#be done in pylearn2
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import E_Step
from pylearn2.models.s3c import Grad_M_Step

#included to make old pkl files load
Split_E_Step = E_Step



#
# ============= Implementation of linear CG based E step
#


from pylearn2.optimization.linear_cg import linear_cg
from theano import config
from pylearn2.models.s3c import damp
from pylearn2.utils import as_floatX
from pylearn2.utils import sharedX
import numpy as np
import theano.tensor as T
from theano import function
from theano import scan
from pylearn2.models.s3c import reflection_clip
from theano.gof.op import get_debug_values, debug_error_message
import warnings





class E_Step_CG(E_Step):
    """ An E Step for the S3C class that updates S_hat using linear conjugate gradient, using the R operator to perform Hessian vector products """

    def __init__(self, h_new_coeff_schedule,
                       s_max_iters,
                       monitor_kl = False,
                       monitor_em_functional = False):
        """Parameters
        --------------
        h_new_coeff_schedule:
            list of coefficients to put on the new value of h on each damped fixed point step
                    (coefficients on s are driven by a special formula)
            length of this list determines the number of fixed point steps
            if None, assumes that the model is not meant to run on its own (ie a larger model
                will specify how to do inference in this layer)
        s_max_iters:
                schedule of max_iters arguments to linear_cg
                    must have same length as the one for h_new_coeff_schedule
        """

        self.autonomous = True

        if h_new_coeff_schedule is None:
            self.autonomous = False
            assert monitor_em_functional is None
        else:
            assert len(s_max_iters) == len(h_new_coeff_schedule)

        self.s_max_iters = s_max_iters

        self.h_new_coeff_schedule = h_new_coeff_schedule
        self.monitor_kl = monitor_kl
        self.monitor_em_functional = monitor_em_functional

        self.model = None

    def infer_S_hat(self, V, H_hat, S_hat, var_s0_hat, var_s1_hat, max_iters):

        alpha = self.model.alpha

        obj = self.truncated_KL( V = V,
                obs = locals() ).mean()

        new_S_hat = linear_cg( fn = obj, params = S_hat, max_iters = max_iters)

        return new_S_hat

    def variational_inference(self, V, return_history = False):
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

        if not self.autonomous:
            raise ValueError("Non-autonomous model asked to perform inference on its own")

        alpha = self.model.alpha

        var_s0_hat = 1. / alpha
        var_s1_hat = self.var_s1_hat()

        H_hat   =    self.init_H_hat(V)
        S_hat =    self.init_S_hat(V)

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
                    'H_hat' : H_hat,
                    'S_hat' : S_hat,
                    'var_s0_hat' : var_s0_hat,
                    'var_s1_hat': var_s1_hat,
                    }

        history = [ make_dict() ]

        count = 2

        for new_H_coeff, max_iters in zip(self.h_new_coeff_schedule, self.s_max_iters):

            S_hat = self.infer_S_hat(V, H_hat, S_hat, var_s0_hat, var_s1_hat, max_iters)

            new_H = self.infer_H_hat(V, H_hat, S_hat, count)
            count += 1

            H_hat = damp(old = H_hat, new = new_H, new_coeff = new_H_coeff)

            check_H(H_hat,V)

            history.append(make_dict())

        if return_history:
            return history
        else:
            return history[-1]

class E_Step_CG_Scan(E_Step):
    """ An E Step for the S3C class that updates S_hat using linear conjugate gradient, using the R operator to perform Hessian vector products """

    def __init__(self, h_new_coeff_schedule,
                       s_max_iters,
                       monitor_kl = False,
                       monitor_em_functional = False,
                       monitor_s_mag = False):
        """Parameters
        --------------
        h_new_coeff_schedule:
            list of coefficients to put on the new value of h on each damped fixed point step
                    (coefficients on s are driven by a special formula)
            length of this list determines the number of fixed point steps
            if None, assumes that the model is not meant to run on its own (ie a larger model
                will specify how to do inference in this layer)
        s_max_iters:
                schedule of max_iters arguments to linear_cg
                    must have same length as the one for h_new_coeff_schedule
        """

        self.autonomous = True

        if h_new_coeff_schedule is None:
            self.autonomous = False
            assert monitor_em_functional is None
        else:
            assert len(s_max_iters) == len(h_new_coeff_schedule)

        self.s_max_iters = s_max_iters

        self.h_new_coeff_schedule = h_new_coeff_schedule
        self.monitor_kl = monitor_kl
        self.monitor_em_functional = monitor_em_functional

        self.monitor_s_mag = monitor_s_mag

        self.model = None

    def infer_S_hat(self, V, H_hat, S_hat, var_s0_hat, var_s1_hat, max_iters):

        alpha = self.model.alpha

        obj = self.truncated_KL( V = V,
                obs = locals() ).mean()

        new_S_hat = linear_cg( fn = obj, params = S_hat, max_iters = max_iters)

        return new_S_hat

    def variational_inference(self, V, return_history = False):
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

        if not self.autonomous:
            raise ValueError("Non-autonomous model asked to perform inference on its own")

        alpha = self.model.alpha

        var_s0_hat = 1. / alpha
        var_s1_hat = self.infer_var_s1_hat()

        H_hat   =    self.init_H_hat(V)
        S_hat =    self.init_S_hat(V)

        def inner_function(new_H_coeff, max_iters, H_hat, S_hat):

            S_hat = self.infer_S_hat(V, H_hat, S_hat, var_s0_hat, var_s1_hat, max_iters)

            new_H = self.infer_H_hat(V, H_hat, S_hat)

            H_hat = damp(old = H_hat, new = new_H, new_coeff = new_H_coeff)

            return H_hat, S_hat


        (H_hats, S_hats), _ = scan( fn = inner_function, sequences =
                [T.constant(np.cast[config.floatX](np.asarray(self.h_new_coeff_schedule))),
                 T.constant(np.asarray(self.s_max_iters))],
                                        outputs_info = [ H_hat, S_hat ] )

        if return_history:
            hist =  [
                    {'H_hat' : H_hats[i],
                     'S_hat' : S_hats[i],
                     'var_s0_hat' : var_s0_hat,
                     'var_s1_hat' : var_s1_hat
                    } for i in xrange(len(self.h_new_coeff_schedule)) ]

            hist.insert(0, { 'H_hat' : H_hat,
                             'S_hat' : S_hat,
                             'var_s0_hat' : var_s0_hat,
                             'var_s1_hat' : var_s1_hat
                            } )
            return hist

        return {
                'H_hat' : H_hats[-1],
                'S_hat' : S_hats[-1],
                'var_s0_hat' : var_s0_hat,
                'var_s1_hat': var_s1_hat,
                }


class E_Step_CG_Scan_StartOn(E_Step_CG_Scan):

    #def __init__(self, ** kwargs):
    #    warnings.warn("need to put a safe call here")

    #    super(E_Step_CG_Scan_StartOn, self).__init__(**kwargs)


    def init_H_hat(self, V):

        if self.model.recycle_q:
            rval = self.model.prev_H
            if config.compute_test_value != 'off':
                if rval.get_value().shape[0] != V.tag.test_value.shape[0]:
                    raise Exception('E step given wrong test batch size', rval.get_value().shape, V.tag.test_value.shape)
        else:
            rval = T.alloc(1., V.shape[0], self.model.nhid)

            for rval_value, V_value in get_debug_values(rval, V):
                if rval_value.shape[0] != V_value.shape[0]:
                    debug_error_message("rval.shape = %s, V.shape = %s, element 0 should match but doesn't", str(rval_value.shape), str(V_value.shape))

        return rval


    def init_S_hat(self, V):
        if self.model.recycle_q:
            rval = self.model.prev_S_hat
        else:
            rval = T.dot(V, self.model.W)

        return rval


class S3C_Shared(S3C):
    """A version of S3C where the variational parameters are
        stored as shared variables so that looping in the
        inference algorithm may be implemented in python
        rather than theano. """

    def __init__(self, ** kwargs):

        super(S3C_Shared, self).__init__(**kwargs)

        if self.recycle_q:
            raise NotImplementedError()

    def get_hidden_obs(self, V, return_history = False):

        return self.e_step.get_hidden_obs(return_history)

    def learn_mini_batch(self, X):
        self.e_step.prereq(X)

        super(S3C_Shared, self).learn_mini_batch(X)


class Inferotron:
    def __init__(self, e_step):
        self.e_step = e_step

    def __call__(self, X):
        self.e_step.variational_inference(X)


class E_Step_CG_Shared(E_Step_CG):

    def __init__(self, **kwargs):

        #Configure the E Step
        super(E_Step_CG_Shared, self).__init__(**kwargs)


        #Figure out whether we're in history mode or not
        self.infer_history = self.monitor_kl or self.monitor_em_functional


        self.prereq = Inferotron(self)

    def get_monitoring_channels(self, V):
        d = super(E_Step_CG_Shared, self).get_monitoring_channels(V)

        for key in d:
            d[key] = (d[key], [self.prereq])

        return d

    def get_hidden_obs(self, return_history):

        if self.infer_history:
            if return_history:
                rval = []
                for i in xrange(len(self.H_hat)):
                    rval.append(
                            { 'H_hat': self.H_hat[i],
                              'S_hat': self.S_hat[i],
                              'var_s0_hat': self.var_s0_hat,
                              'var_s1_hat': self.var_s1_hat
                              })
                return rval


            H = self.H_hat[-1]
            S = self.S_hat[-1]
        else:
            assert not return_history
            H = self.H_hat
            S = self.S_hat

        return { 'H_hat': H, 'S_hat' : S,
                'var_s0_hat': self.var_s0_hat,
                'var_s1_hat': self.var_s1_hat
                }


    def register_model(self, model):

        self.model = model

        self.nhid = model.nhid

        self.allocate_shared()
        self.redo_theano()

    def redo_theano(self):
        """re-compiles all theano functions"""
        self.make_init_func()
        self.make_update_funcs()


    def make_init_func(self):
        """ compiles theano function for initializing inference """

        V = T.matrix("V")

        alpha = self.model.alpha

        var_s0_hat = 1. / alpha
        var_s1_hat = self.infer_var_s1_hat()

        H_hat = self.init_H_hat(V)
        S_hat = self.init_S_hat(V)

        if self.infer_history:
            H_targ = self.H_hat[0]
            S_targ = self.S_hat[0]
        else:
            H_targ = self.H_hat
            S_targ = self.S_hat

        updates = {
                self.var_s0_hat : var_s0_hat,
                self.var_s1_hat : var_s1_hat,
                H_targ : H_hat,
                S_targ : S_hat
                }

        print "Compiling init_func..."
        self.init_func = function([V], updates = updates)
        print "...done"

    def allocate_shared(self):
        """ allocates the shared variables the E Step needs """

        self.var_s0_hat = sharedX(np.zeros((self.nhid,)),name='var_s0_hat')
        self.var_s1_hat = sharedX(np.zeros((self.nhid,)),name='var_S1_hat')

        if self.infer_history:
            self.H_hat = []
            self.S_hat = []

            for i in xrange(len(self.h_new_coeff_schedule)+1):
                self.H_hat.append(sharedX(np.zeros((1,self.nhid)),name='H_hat_'+str(i+1)))
                self.S_hat.append(sharedX(np.zeros((1,self.nhid)),name='S_hat_'+str(i+1)))
        else:
            self.H_hat = sharedX(np.zeros((1,self.nhid)),name="H_hat")
            self.S_hat = sharedX(np.zeros((1,self.nhid)),name="S_hat")



    def make_update_funcs(self):

        if self.infer_history:
            self.update_funcs = []
            for i in xrange(len(self.h_new_coeff_schedule)):
                print 'compiling update ',i,'...'
                f = self.get_update_func(
                        H_in = self.H_hat[i],
                        S_in = self.S_hat[i],
                        H_out = self.H_hat[i+1],
                        S_out = self.S_hat[i+1])
                print '...done'
                self.update_funcs.append(f)
        else:
            print 'compiling update func...'
            self.update_func = self.get_update_func(
                    H_in = self.H_hat,
                    S_in = self.S_hat,
                    H_out = self.H_hat,
                    S_out = self.S_hat)
            print '...done'



    def get_update_func(self, H_in, S_in, H_out, S_out):

        V = T.matrix('V')
        iters = T.iscalar('max_iters')
        new_coeff = T.scalar('h_new_coeff')

        S_hat = self.infer_S_hat(V, H_in, S_in, self.var_s0_hat, self.var_s1_hat, iters)

        H_hat = self.infer_H_hat(V, H_in, S_hat)

        H_hat = damp(old = H_in, new = H_hat, new_coeff = new_coeff)

        updates = { H_out : H_hat, S_out : S_hat }

        f = function( [V,iters,new_coeff], updates = updates)

        return f

    def variational_inference(self, X):
        """
            TODO: WRITEME
        """

        if not self.autonomous:
            raise ValueError("Non-autonomous model asked to perform inference on its own")

        self.init_func(X)


        for i in xrange(len(self.h_new_coeff_schedule)):
            new_H_coeff = self.h_new_coeff_schedule[i]
            max_iters = self.s_max_iters[i]

            if self.infer_history:
                f = self.update_funcs[i]
            else:
                f = self.update_func

            f(X,max_iters,new_H_coeff)


