from pylearn2.utils import serial
from pylearn2.datasets import cifar10
from pylearn2.datasets import preprocessing

train = cifar10.CIFAR10(which_set="train")

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(8,8),num_patches=2000000))
pipeline.items.append(preprocessing.GlobalContrastNormalization(std_bias = 0.0, use_norm = 1.))

test = cifar10.CIFAR10(which_set="test")


train.apply_preprocessor(preprocessor = pipeline, can_fit = True)
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)



train.use_design_loc('/data/lisatmp/goodfeli/cifar10_sphere_train_2M_design.npy')
test.use_design_loc('/data/lisatmp/goodfeli/cifar10_sphere_test_2M_design.npy')


serial.save('/data/lisatmp/goodfeli/cifar10_sphere_train_2M.pkl',train)
serial.save('/data/lisatmp/goodfeli/cifar10_sphere_test_2M.pkl',test)

train = serial.load('/data/lisatmp/goodfeli/cifar10_sphere_train_2M.pkl')
