from pylearn2.utils import serial
from pylearn2.datasets import cifar10
from pylearn2.datasets import preprocessing
import os

goodfeli_tmp = os.environ['GOODFELI_TMP']

train = cifar10.CIFAR10(which_set="train")

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(8,8),num_patches=2000000))
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA())

test = cifar10.CIFAR10(which_set="test")

train.apply_preprocessor(preprocessor = pipeline, can_fit = True)
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)


train.use_design_loc(goodfeli_tmp + '/cifar10_preprocessed_train_2M_design.npy')
test.use_design_loc(goodfeli_tmp + '/cifar10_preprocessed_test_2M_design.npy')


serial.save(goodfeli_tmp + '/cifar10_preprocessed_train_2M.pkl',train)
serial.save(goodfeli_tmp + '/cifar10_preprocessed_test_2M.pkl',test)

serial.save(goodfeli_tmp + '/cifar10_preprocessed_pipeline_2M.pkl',pipeline)
