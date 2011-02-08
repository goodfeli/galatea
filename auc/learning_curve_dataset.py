import pylearn.datasets.utlc as pdu
import numpy as N

class learning_curve_dataset:
    def __init__(self, data_name):
        X0 = {}
        X0['devel'], X0['valid'], X0['final'] = pdu.load_ndarray_dataset(data_name, normalize=False, transfer = False)
       
        self.X0 = X0


        Y = {}
        if data_name == 'ule':
            Y['devel'], Y['valid'], Y['final'] = pdu.load_ndarray_label(data_name)
        #endif
        self.Y = Y

        #deshuffle the devel set
        rng = N.random.RandomState([1,2,3])
        perm = rng.permutation(self.X0['devel'].shape[0])
        
        self.X0['devel'][perm,:] = self.X0['devel'][N.asarray(range(self.X0['devel'].shape[0])),:]
        self.Y['devel'][perm] = self.Y['devel']


        #X0[devel] matches matlab code here
        #print 'X0[devel] sum'
        #print self.X0['devel'].sum()
        #print self.X0['devel'][0:2,380:390]


        #for Y in self.Y.values():
        #    print 'Y shape'
        #    print Y.shape
        #    print 'Y checksum'
        #    print N.dot(N.asarray(range(Y.shape[0])),Y)
        #



        self.quantized = None #not sure how this gets set in the matlab code
        self.kernelized = None #not sure how this gets set in the matlab code
    #close def__init__
#close class learning_curve_dataset

