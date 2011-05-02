from framework.utils import serial
from framework.datasets import cifar10
from framework.datasets import preprocessing

train = cifar10.CIFAR10(which_set="train")

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(8,8),num_patches=2000000))
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA())

test = cifar10.CIFAR10(which_set="test")

train.apply_preprocessor(preprocessor = pipeline, can_fit = True)
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)


train.use_design_loc('/data/lisatmp/goodfeli/cifar10_preprocessed_train_2M_design.npy')
test.use_design_loc('/data/lisatmp/goodfeli/cifar10_preprocessed_test_2M_design.npy')


serial.save('cifar10_preprocessed_train_2M.pkl',train)
serial.save('cifar10_preprocessed_test_2M.pkl',test)


