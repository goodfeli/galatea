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

serial.save('cifar10_preprocessor_2M.pkl',train)


