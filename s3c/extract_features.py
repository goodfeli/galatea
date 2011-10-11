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
import sys
config.floatX = 'float32'

class FeatureExtractor:
    def __init__(self, batch_size, model_path, pooling_region_counts,
           save_paths, feature_type, dataset_name, which_set, restrict = None):
        self.batch_size = batch_size
        self.model_path = model_path
        self.restrict = restrict

        assert len(pooling_region_counts) == len(save_paths)

        self.pooling_region_counts = pooling_region_counts
        self.save_paths = save_paths
        self.feature_type = feature_type
        self.which_set = which_set
        self.dataset_name = dataset_name

    def __call__(self):
        model_path = self.model_path
        batch_size = self.batch_size
        feature_type = self.feature_type
        pooling_region_counts = self.pooling_region_counts
        dataset_name = self.dataset_name

        stl10 =  dataset_name == 'stl10'
        cifar10 = dataset_name == 'cifar10'

        assert stl10 or cifar10

        print 'loading model'
        model = serial.load(model_path)
        model.set_dtype('float32')

        print 'loading dataset'
        if stl10:
            dataset = serial.load('${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/'+self.which_set+'.pkl')
            train_size = 5000
            test_size  = 8000
        elif cifar10:
            from pylearn2.datasets.cifar10 import CIFAR10
            dataset = CIFAR10(which_set = self.which_set)
            train_size = 50000
            test_size  = 10000


        full_X = dataset.get_design_matrix()
        num_examples = full_X.shape[0]

        if self.restrict is not None:
            print 'restricting to examples ',self.restrict[0],' through ',self.restrict[1],' exclusive'
            full_X = full_X[self.restrict[0]:self.restrict[1],:]





        if self.which_set == 'train':
                assert num_examples == train_size
        elif self.which_set == 'test':
            assert num_examples == test_size

        else:
            assert False

        #update for after restriction
        num_examples = full_X.shape[0]

        dataset.X = None
        dataset.design_loc = None
        dataset.compress = False

        size = np.sqrt(model.nvis/3)

        patchifier = ExtractGridPatches( patch_shape = (size,size), patch_stride = (1,1) )

        if size ==6:
            if stl10:
                pipeline = serial.load('${PYLEARN2_DATA_PATH}/stl10/stl10_patches/preprocessor.pkl')
            elif cifar10:
                pipeline = serial.load('${GOODFELI_TMP}/cifar10_preprocessed_pipeline_2M_6x6.pkl')
        else:
            print size
            assert False

        assert isinstance(pipeline.items[0], ExtractPatches)
        pipeline.items[0] = patchifier


        print 'defining features'
        V = T.matrix()
        model.make_Bwp()
        d = model.e_step.mean_field(V = V)

        H = d['H']
        Mu1 = d['Mu1']

        assert H.dtype == 'float32'
        assert Mu1.dtype == 'float32'

        if self.feature_type == 'map_hs':
            feat = (H > 0.5) * Mu1
        elif self.feature_type == 'exp_hs':
            feat = H * Mu1
        elif self.feature_type == 'exp_h':
            feat = H
        else:
            raise NotImplementedError()

        assert feat.dtype == 'float32'
        print 'compiling theano function'
        f = function([V],feat)

        topo_feat_var = T.TensorType(broadcastable = (False,False,False,False), dtype='float32')()
        region_features = function([topo_feat_var],
                topo_feat_var.mean(axis=(1,2)) )

        def average_pool( stride ):
            def point( p ):
                return p * ns / stride

            rval = np.zeros( (topo_feat.shape[0], stride, stride, topo_feat.shape[3] ) , dtype = 'float32')

            for i in xrange(stride):
                for j in xrange(stride):
                    rval[:,i,j,:] = region_features( topo_feat[:,point(i):point(i+1), point(j):point(j+1),:] )

            return rval

        outputs = [ np.zeros((num_examples,count,count,model.nhid),dtype='float32') for count in pooling_region_counts ]

        fd = DenseDesignMatrix(X = np.zeros((1,1),dtype='float32'), view_converter = DefaultViewConverter([1, 1, model.nhid] ) )

        ns = 32 - size + 1
        depatchifier = ReassembleGridPatches( orig_shape  = (ns, ns), patch_shape=(1,1) )

        for i in xrange(0,num_examples-batch_size+1,batch_size):
            print i
            t1 = time.time()

            d = copy.copy(dataset)
            d.set_design_matrix(full_X[i:i+batch_size,:])

            t2 = time.time()

            #print '\tapplying preprocessor'
            d.apply_preprocessor(pipeline, can_fit = False)
            X2 = d.get_design_matrix()

            t3 = time.time()

            #print '\trunning theano function'
            feat = f(X2)

            t4 = time.time()

            assert feat.dtype == 'float32'

            feat_dataset = copy.copy(fd)
            feat_dataset.set_design_matrix(feat)

            #print '\treassembling features'
            feat_dataset.apply_preprocessor(depatchifier)

            #print '\tmaking topological view'
            topo_feat = feat_dataset.get_topological_view()
            assert topo_feat.shape[0] == batch_size

            t5 = time.time()

            #average pooling
            for output, count in zip(outputs, pooling_region_counts):
                output[i:i+batch_size,...] = average_pool(count)

            t6 = time.time()

            print (t6-t1, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)

        for output, save_path in zip(outputs, self.save_paths):
            np.save(save_path,output)

if __name__ == '__main__':
    assert len(sys.argv) == 2
    yaml_path = sys.argv[1]

    assert yaml_path.endswith('.yaml')
    val = yaml_path[0:-len('.yaml')]
    os.environ['FEATURE_EXTRACTOR_YAML_PATH'] = val
    os.putenv('FEATURE_EXTRACTOR_YAML_PATH',val)

    extractor = yaml_parse.load_path(yaml_path)

    extractor()
