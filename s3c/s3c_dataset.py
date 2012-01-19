from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
import theano.tensor as T
from theano import function
import numpy as np

class S3C_Dataset(DenseDesignMatrix):

    def __init__(self, raw, transformer):
        self.dataset = raw
        self.transformer = transformer

        self.transformer.make_pseudoparams()

        V = T.matrix()

        obs = self.transformer.get_hidden_obs(V)

        H = obs['H_hat']
        S = obs['S_hat']

        F = H * S

        print 'compiling transformer...'
        self.transform_func = function([V], F)
        print '...done'


        N = self.transformer.nhid

        r = int(np.sqrt(N))
        c = N / r

        if N == r * c:
            shape = (r,c,1)
        else:
            shape = (N,1,1)


        self.view_converter = DefaultViewConverter(shape=shape)


    def get_batch_design(self, batch_size):

        X = self.dataset.get_batch_design(batch_size)

        return self.transform_func(X)

    def weights_view_shape(self):
        n = self.transformer.nvis / 3

        h = int(np.sqrt(n))

        w = n / h

        assert h * w == n

        return (h,w,3)

    def get_weights_view(self, mat):

        recons = np.dot(mat, self.transformer.W.get_value().T)

        return self.dataset.get_topological_view(recons)
