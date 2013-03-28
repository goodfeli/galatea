from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets.transformer_dataset import TransformerDataset
import theano.tensor as T
from theano import function
import numpy as np

#TransformerDataset goes first so its get_batch_design will go first
class S3C_Dataset(TransformerDataset,DenseDesignMatrix):

    def __init__(self, raw, transformer, cpu_only = False):
        TransformerDataset.__init__(self,raw, transformer, cpu_only = cpu_only)

        N = self.transformer.nhid

        r = int(np.sqrt(N))
        c = N / r

        if N == r * c:
            shape = (r,c,1)
        else:
            shape = (N,1,1)

        self.view_converter = DefaultViewConverter(shape=shape)

    def weights_view_shape(self):
        n = self.transformer.nvis / 3

        h = int(np.sqrt(n))

        w = n / h

        assert h * w == n

        return (h,w,3)

    def get_weights_view(self, mat):

        recons = np.dot(mat * self.transformer.mu.get_value(), self.transformer.W.get_value().T)

        return self.raw.get_topological_view(recons)

    def get_topological_view(self, mat):
        print 'mat: '+str((mat.min(),mat.mean(),mat.max()))
        rval =  self.get_weights_view(mat)
        print 'rval: '+str((rval.min(),rval.mean(),rval.max()))
        return rval

    def get_unprocessed_batch_design(self, batch_size):
        return self.raw.get_batch_design(batch_size)

    def get_batch_topo(self, batch_size):
        return DenseDesignMatrix.get_batch_topo(self, batch_size)
