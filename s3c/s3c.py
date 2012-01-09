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

class E_Step_CG(E_Step):

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
