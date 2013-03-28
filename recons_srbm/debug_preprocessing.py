from framework.utils import serial
from framework.datasets import cifar10
from framework.datasets import preprocessing
import numpy as N

train = cifar10.CIFAR10(which_set="train")

train.apply_preprocessor(preprocessing.ExtractPatches(patch_shape=(8,8),num_patches=150000))

orig_patches = train.get_topological_view().copy()
print (orig_patches.min(),orig_patches.max())
#orig_patches -= 0.5
orig_patches -= 127.5
orig_patches /= N.abs(orig_patches).max()
print (orig_patches.min(),orig_patches.max())

train.apply_preprocessor(preprocessing.GlobalContrastNormalization(std_bias=10.))


processed_patches = train.get_topological_view().copy()
processed_patches /= N.abs(processed_patches).max()

train.apply_preprocessor(preprocessing.ZCA(), can_fit = True)

zca_patches = train.get_topological_view().copy()
zca_patches /= N.abs(zca_patches).max()

print id(zca_patches)
print id(processed_patches)
print N.abs(zca_patches-processed_patches).mean()

concat = N.concatenate((orig_patches,processed_patches,zca_patches),axis=2)

train.set_topological_view(concat)


serial.save('debug.pkl',train)


