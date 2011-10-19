from pylearn2.utils import serial
from pylearn2.datasets.cifar100 import CIFAR100
from pylearn2.datasets.tl_challenge import TL_Challenge
from pylearn2.datasets import preprocessing
import os
import numpy as np

goodfeli_tmp = os.environ['GOODFELI_TMP']

train = CIFAR100(which_set="train")

aug = TL_Challenge(which_set="unlabeled")
aug2 = TL_Challenge(which_set="train")

train.set_design_matrix(np.concatenate( (train.X, aug.X, aug2.X), axis=0))

del aug
del aug2

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(6,6),num_patches=2000000))
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA())


train.apply_preprocessor(preprocessor = pipeline, can_fit = True)


train.use_design_loc(goodfeli_tmp + '/tl_challenge_patches_2M_6x6_design.npy')


serial.save(goodfeli_tmp + '/tl_challenge_patches_2M_6x6.pkl',train)

serial.save(goodfeli_tmp + '/tl_challenge_patches_2M_6x6_prepro.pkl',pipeline)
