from pylearn2.datasets.norb_small import NORBSmall
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.image import rescale
import numpy as np
import time

class NORB_Tiny(DenseDesignMatrix):

    def __init__(self, which_set):

        print 'making tiny norb...'
        t1 = time.time()
        small = NORBSmall(which_set)

        t = small.get_topological_view(small.X)[:,:,:,0:1] / 255.

        t = t[:,10:86,10:86,:]

        r,c = 20, 20

        new_t = np.zeros( (t.shape[0], r, c, 1 ), dtype = t.dtype )

        for i in xrange(t.shape[0]):
            print i
            new_t[i,:,:,:] = rescale(t[i,:], (r,c) )

        new_t -= 0.5

        super(NORB_Tiny, self).__init__(topo_view = new_t)

        t2 = time.time()

        print '...took '+str(t2-t1)+' seconds'
