from pylearn2.utils import serial
from pylearn2.datasets import cifar10
from pylearn2.datasets import preprocessing

train = cifar10.CIFAR10(which_set="train")

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(8,8),num_patches=150000))
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA())

test = cifar10.CIFAR10(which_set="test")

train.apply_preprocessor(preprocessor = pipeline, can_fit = True)
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)


train.use_design_loc('/data/lisatmp/goodfeli/cifar10_preprocessed_train_design.npy')
test.use_design_loc('/data/lisatmp/goodfeli/cifar10_preprocessed_test_design.npy')


serial.save('cifar10_preprocessed_train.pkl',train)
serial.save('cifar10_preprocessed_test.pkl',test)


