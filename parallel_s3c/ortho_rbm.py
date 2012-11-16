from pylearn2.models import Model
import numpy as np
from pylearn2.utils import sharedX


class OrthoRBM(Model):

    def __init__(self, nvis, nhid, irange, init_bias_hid, init_beta, init_scale):

        self.scale = sharedX(np.zeros((nvis,))+init_scale)
        self.W = sharedX(random_ortho_columns(nvis, nhid))
        self.bias_hid = sharedX(np.zeros((nhid,))+init_bias_hid)
