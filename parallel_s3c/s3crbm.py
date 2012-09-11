from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import E_Step


class S3CRBM(S3C):
	"""

        An S3C model constrained to have
        W^T beta W be 0 on the off-diagonal entries,
        making the model also be an ssRBM.

        This means that inference can be done in a single pass.

        There are two ways of doing inference, based on different
        ways of interpreting the constraint:

            1. We use the RBM update equations. This is motivated
            by assuming the constraint holds with equality.

            2. We do one pass of the S3C update equations. This is
            motivated by assuming the interaction terms are small.
            *Unlike in standard S3C, we update h and s simultaneously
            so we don't actually use the same update equations *
    """


class RBM_E_Step(E_Step):
    """ An E step that works by using the RBM update equations """

    def __init__(self,
        monitor_kl = False,
        monitor_energy_functional = False,
        monitor_s_mag = False,
        monitor_ranges = False):

        super(RBM_E_Step, self).__init__(
                h_new_coeff_schedule = [],
                s_new_coeff_schedule = [],
                monitor_kl = monitor_kl,
                monitor_energy_functional = monitor_energy_functional,
                monitor_s_mag = monitor_s_mag,
                monitor_ranges = monitor_ranges)

        def init_H_hat(self, V):
            raise TypeError("RBM_E_Step doesn't need to initialize an optimization procedure")

        def init_S_hat(self, V):
            raise TypeError("RBM_E_Step doesn't need to initialize an optimization procedure")

        TODO--finish subclassing the rest of the methods.


class S3C_One_Shot_E_Step(E_Step):
    """ An E step that works by running one pass of the S3C update
    equations. We assume the interaction terms are small so it converges
    in one step. """

