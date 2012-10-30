from pylearn2.utils import serial
from pylearn2.datasets import cifar10
from pylearn2.datasets import preprocessing
import os

goodfeli_tmp = os.environ['GOODFELI_TMP']

train = cifar10.CIFAR10(which_set="train")

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(6,6),num_patches=2000000))
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA())

test = cifar10.CIFAR10(which_set="test")

train.apply_preprocessor(preprocessor = pipeline, can_fit = True)
print 'processing test set'
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)


print 'saving'
train.use_design_loc(goodfeli_tmp + '/cifar10_preprocessed_train_2M_6x6_design.npy')
test.use_design_loc(goodfeli_tmp + '/cifar10_preprocessed_test_2M_6x6_design.npy')


serial.save(goodfeli_tmp + '/cifar10_preprocessed_train_2M_6x6.pkl',train)
print 'done saving train'
serial.save(goodfeli_tmp + '/cifar10_preprocessed_test_2M_6x6.pkl',test)
print 'done saving test'

serial.save(goodfeli_tmp + '/cifar10_preprocessed_pipeline_2M_6x6.pkl',pipeline)
print 'done saving pipeline'
