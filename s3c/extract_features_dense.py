import os
from pylearn2.config import yaml_parse
import warnings
import time
import copy
import numpy as np
from theano import config
from theano import tensor as T
#from theano.sandbox.neighbours import images2neibs
from theano import function
from pylearn2.datasets.preprocessing import ExtractPatches, ExtractGridPatches, ReassembleGridPatches
from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.cifar100 import CIFAR100
from pylearn2.datasets.tl_challenge import TL_Challenge
import sys
config.floatX = 'float32'

class halver:
    def __init__(self,f,nhid):
        self.f = f
        self.nhid = nhid

    def __call__(self,X):
        m = X.shape[0]
        #mid = m/2
        m1 = m/3
        m2 = 2*m/3

        rval = np.zeros((m,self.nhid),dtype='float32')
        rval[0:m1,:] = self.f(X[0:m1,:])
        rval[m1:m2,:] = self.f(X[m1:m2,:])
        rval[m2:,:] = self.f(X[m2:,:])

        return rval


class FeatureExtractor:
    def __init__(self, batch_size, model_path,
           save_path, feature_type, dataset,
           chunk_size = None, restrict = None):

        if chunk_size is not None and restrict is not None:
            raise NotImplementedError("Currently restrict is used internally to "
                    "implement chunk_size, so a client may not specify both")

        self.batch_size = batch_size
        self.model_path = model_path
        self.restrict = restrict


        self.save_path = save_path
        self.feature_type = feature_type
        self.dataset = dataset
        self.chunk_size = chunk_size

    def __call__(self):

        print 'loading model'
        model_path = self.model_path
        self.model = serial.load(model_path)
        self.model.set_dtype('float32')
        self.size = int(np.sqrt(self.model.nvis/3))

        if self.chunk_size is not None:
            dataset = self.dataset
            num_examples = dataset.X.shape
            assert num_examples % self.chunk_size == 0

            self.chunk_id = 0
            for i in xrange(0,num_examples, self.chunk_size):
                self.restrict = (i, i + self.chunk_size)

                self._execute()

                self.chunk_id += 1
        else:
            self._execute()

    def _execute(self):

        batch_size = self.batch_size
        feature_type = self.feature_type
        dataset_family = self.dataset
        model = self.model
        size = self.size
        save_path = self.save_path

        nan = 0

        dataset = self.dataset

        full_X = dataset.get_design_matrix()
        num_examples = full_X.shape[0]

        if self.restrict is not None:
            assert self.restrict[1]  <= full_X.shape[0]

            print 'restricting to examples ',self.restrict[0],' through ',self.restrict[1],' exclusive'
            full_X = full_X[self.restrict[0]:self.restrict[1],:]

            assert self.restrict[1] > self.restrict[0]

        #update for after restriction
        num_examples = full_X.shape[0]

        assert num_examples > 0

        print 'defining features'
        V = T.matrix('V')
        model.make_pseudoparams()
        d = model.e_step.variational_inference(V = V)

        H = d['H_hat']
        Mu1 = d['S_hat']

        assert H.dtype == 'float32'
        assert Mu1.dtype == 'float32'

        if self.feature_type == 'map_hs':
            feat = (H > 0.5) * Mu1
        elif self.feature_type == 'map_h':
            feat = T.cast(H > 0.5, dtype='float32')
        elif self.feature_type == 'exp_hs':
            feat = H * Mu1
        elif self.feature_type == 'exp_h':
            feat = H
        elif self.feature_type == 'exp_h_thresh':
            feat = H * (H > .01)
        else:
            raise NotImplementedError()



        assert feat.dtype == 'float32'
        print 'compiling theano function'
        f = function([V],feat)

        output = np.zeros((full_X.shape[0], model.nhid),dtype='float32')


        if len(range(0,num_examples-batch_size+1,batch_size)) <= 0:
            print num_examples
            print batch_size

        for i in xrange(0,num_examples-batch_size+1,batch_size):
            t1 = time.time()

            X2 = full_X[i:i+batch_size,:]

            feat = f(X2)

            if np.any(np.isnan(feat)):
                nan += np.isnan(feat).sum()
                feat[np.isnan(feat)] = 0

            output[i:i+batch_size,:] = feat

            t2 = time.time()

            print i,' ',(t2-t1)


        if self.chunk_size is not None:
                assert save_path.endswith('.npy')
                save_path_pieces = save_path.split('.npy')
                assert len(save_path_pieces) == 2
                assert save_path_pieces[1] == ''
                save_path = save_path_pieces[0] + '_' + chr(ord('A')+self.chunk_id)+'.npy'
        np.save(save_path,output)


        if nan > 0:
            warnings.warn(str(nan)+' features were nan')

if __name__ == '__main__':
    assert len(sys.argv) == 2
    yaml_path = sys.argv[1]

    assert yaml_path.endswith('.yaml')
    val = yaml_path[0:-len('.yaml')]
    os.environ['FEATURE_EXTRACTOR_YAML_PATH'] = val
    os.putenv('FEATURE_EXTRACTOR_YAML_PATH',val)

    extractor = yaml_parse.load_path(yaml_path)

    extractor()
