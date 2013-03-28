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


class FeaturesDataset:
    def __init__(self, dataset_maker, num_examples, pipeline_path):
        """
            dataset_maker: A callable that returns a Dataset
            num_examples: the number of examples we expect the dataset to have
                          (just for error checking purposes)
        """
        self.dataset_maker = dataset_maker
        self.num_examples = num_examples
        self.pipeline_path = pipeline_path


stl10 = {}
stl10_no_shelling = {}
stl10_size = { 'train' : 5000, 'test' : 8000 }

cifar10 = {}
cifar10_size = { 'train' : 50000, 'test' : 10000 }

cifar100 = {}
cifar100_size = { 'train' : 50000, 'test' : 10000 }

tl_challenge = {}
tl_challenge_size = { 'train' : 120, 'test' : 0 }

class dataset_loader:
    def __init__(self, path):
        self.path = path

    def __call__(self):
        return serial.load(self.path)

class dataset_constructor:
    def __init__(self, cls, which_set):
        self.cls = cls
        self.which_set = which_set

    def __call__(self):
        return self.cls(self.which_set)


for which_set in ['train', 'test']:
    stl10[which_set] = {}
    stl10_no_shelling[which_set] = {}
    cifar10[which_set] = {}
    cifar100[which_set] = {}
    tl_challenge[which_set] = {}

    #this is for patch size, not datset size
    for size in [6]:
        stl10[which_set][size] = FeaturesDataset( dataset_maker = dataset_loader( '${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/'+which_set+'.pkl'),
                                            num_examples = stl10_size[which_set],
                                            pipeline_path = '${PYLEARN2_DATA_PATH}/stl10/stl10_patches/preprocessor.pkl')

        stl10_no_shelling[which_set][size] = FeaturesDataset( dataset_maker = dataset_loader( '${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/'+which_set+'.pkl'),
                                            num_examples = stl10_size[which_set],
                                            pipeline_path = '${GOODFELI_TMP}/stl10/stl10_patches_no_shelling/preprocessor.pkl')

        cifar10[which_set][size] = FeaturesDataset( dataset_maker = dataset_constructor( CIFAR10, which_set),
                                                num_examples = cifar10_size[which_set],
                                                pipeline_path = '${GOODFELI_TMP}/cifar10_preprocessed_pipeline_2M_6x6.pkl')

        cifar100[which_set][size] = FeaturesDataset( dataset_maker = dataset_constructor( CIFAR100, which_set),
                                                num_examples = cifar100_size[which_set],
                                                pipeline_path = '${PYLEARN2_DATA_PATH}/cifar100/cifar100_patches/preprocessor.pkl')

        tl_challenge[which_set][size] = FeaturesDataset( dataset_maker = dataset_constructor( TL_Challenge, which_set),
                                                num_examples = tl_challenge_size[which_set],
                                                pipeline_path = '${GOODFELI_TMP}/tl_challenge_patches_2M_6x6_prepro.pkl')


    stl10[which_set][8] = FeaturesDataset( dataset_maker = dataset_loader( '${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/'+which_set+'.pkl'),
                                            num_examples = stl10_size[which_set],
                                            pipeline_path = '${PYLEARN2_DATA_PATH}/stl10/stl10_patches_8x8/preprocessor.pkl')

class FeatureExtractor:
    def __init__(self, model, preprocessor,
            pooling_region_counts = [3],
           feature_type = 'exp_h'):

        self.pooling_region_counts = pooling_region_counts
        self.feature_type = feature_type

        self.model = model
        self.size = int(np.sqrt(self.model.nvis/3))
        self.model.set_dtype('float32')
        self.preprocessor = preprocessor

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

        if config.device.startswith('gpu') and model.nhid >= 4000:
            f = halver(f, model.nhid)

        self.f = f

        topo_feat_var = T.TensorType(broadcastable = (False,False,False,False), dtype='float32')()
        self.region_features = function([topo_feat_var],
                topo_feat_var.mean(axis=(1,2)) )

    def __call__(self, full_X):

        feature_type = self.feature_type
        pooling_region_counts = self.pooling_region_counts
        model = self.model
        size = self.size

        nan = 0


        full_X = full_X.reshape(1,full_X.shape[0],full_X.shape[1],full_X.shape[2])

        if full_X.shape[3] == 1:
            full_X = np.concatenate( (full_X, full_X, full_X), axis=3)

        print 'full_X.shape: '+str(full_X.shape)

        num_examples = full_X.shape[0]
        assert num_examples == 1

        pipeline = self.preprocessor


        def average_pool( stride ):
            def point( p ):
                return p * ns / stride

            rval = np.zeros( (topo_feat.shape[0], stride, stride, topo_feat.shape[3] ) , dtype = 'float32')

            for i in xrange(stride):
                for j in xrange(stride):
                    rval[:,i,j,:] = self.region_features( topo_feat[:,point(i):point(i+1), point(j):point(j+1),:] )

            return rval

        outputs = [ np.zeros((num_examples,count,count,model.nhid),dtype='float32') for count in pooling_region_counts ]

        assert len(outputs) > 0

        fd = DenseDesignMatrix(X = np.zeros((1,1),dtype='float32'), view_converter = DefaultViewConverter([1, 1, model.nhid] ) )

        ns = 32 - size + 1
        depatchifier = ReassembleGridPatches( orig_shape  = (ns, ns), patch_shape=(1,1) )

        batch_size = 1

        for i in xrange(0,num_examples-batch_size+1,batch_size):
            print i
            t1 = time.time()

            d = DenseDesignMatrix( topo_view =  np.cast['float32'](full_X[i:i+batch_size,:]), view_converter = DefaultViewConverter((32,32,3)))

            t2 = time.time()

            #print '\tapplying preprocessor'
            d.apply_preprocessor(pipeline, can_fit = False)
            X2 = d.get_design_matrix()

            t3 = time.time()

            #print '\trunning theano function'
            feat = self.f(X2)

            t4 = time.time()

            assert feat.dtype == 'float32'

            feat_dataset = copy.copy(fd)

            if np.any(np.isnan(feat)):
                nan += np.isnan(feat).sum()
                feat[np.isnan(feat)] = 0

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

        return outputs[0]

if __name__ == '__main__':


    report = open('report.txt','w')


    model = serial.load('/data/lisatmp/goodfeli/darpa_s3c.pkl')
    preprocessor = serial.load('/data/lisatmp/goodfeli/darpa_imagenet_patch_6x6_train_preprocessor.pkl')
    patchifier = ExtractGridPatches( patch_shape = (size,size), patch_stride = (1,1) )
    preprocessor.items.insert(0,patchifier)

    extractor = FeatureExtractor( model = model, preprocessor = preprocessor)

    xavier = '/data/lisatmp/glorotxa/val'
    thumbnail = '/data/lisatmp/goodfeli/darpa_imagenet_valid_thumb'
    feature = '/data/lisatmp/goodfeli/darpa_imagenet_valid_features'

    from galatea.darpa_imagenet.utils import explore_images

    for img_path in explore_images(xavier,'.JPEG'):
        print img_path
        thumbnail_path = img_path.replace(xavier,thumbnail)
        thumbnail_path = thumbnail_path.replace('.JPEG','.npy')
        if os.path.exists(thumbnail_path):
            feature_path = thumbnail_path.replace(thumbnail,feature)
            if not os.path.exists(feature_path):
                print 'making '+feature_path
                X = np.load(thumbnail_path)
                X = extractor(X)
                np.save(feature_path,X)
        else:
            print 'No thumbnail!'
            report.write(img_path)

