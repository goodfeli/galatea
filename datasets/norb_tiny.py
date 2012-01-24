from pylearn2.datasets.norb_small import NORBSmall
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.preprocessing import ExtractGridPatches, ReassembleGridPatches
import numpy as np
import time

class NORB_Tiny(DenseDesignMatrix):

    def __init__(self, which_set):

        print 'making tiny norb...'
        t1 = time.time()
        small = NORBSmall(which_set)

        t = small.get_topological_view(small.X)[:,:,0:1]

        del small

        super(NORB_Tiny, self).__init__(topo = t)

        self.apply_preprocessor(ExtractGridPatches((2,2),(2,2)))

        X = self.X

        self.X = np.zeros((X.shape[0],1),dtype=X.dtype)

        X[:,0] = X.mean(axis=1)

        self.apply_preprocessor(ReassembleGridPatches((16,16),(1,1)))

        t2 = time.time()

        print '...took '+str(t2-t1)+' seconds'
