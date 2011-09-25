from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
import os
from pylearn2.utils import string, serial
import numpy as np

data_dir = string.preprocess('${PYLEARN2_DATA_PATH}')

print 'Loading STL10-10 unlabeled and train datasets...'
downsampled_dir = data_dir + '/stl10/stl10_32x32'

data = serial.load(downsampled_dir + '/unlabeled.pkl')
supplement = serial.load(downsampled_dir + '/train.pkl')

print 'Concatenating datasets...'
data.set_design_matrix(np.concatenate((data.X,supplement.X),axis=0))
del supplement

print "Preprocessing the data..."
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(6,6),num_patches=2*1000*1000))
#pipeline.items.append(preprocessing.GlobalContrastNormalization())
#pipeline.items.append(preprocessing.ZCA())
data.apply_preprocessor(preprocessor = pipeline, can_fit = True)

print data.X.mean()

#data.use_design_loc(patch_dir + '/data.npy')
#serial.save(patch_dir + '/data.pkl',data)
#serial.save(patch_dir + '/preprocessor.pkl',pipeline)
