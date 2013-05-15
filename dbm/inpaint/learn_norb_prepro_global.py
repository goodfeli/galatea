from pylearn2.datasets.norb_small import FoveatedNORB

dataset = FoveatedNORB(which_set='train')

from pylearn2.datasets.preprocessing import Standardize

standardize = Standardize(global_mean=True, global_std=True)

standardize.apply(dataset, can_fit=True)

from pylearn2.utils import serial
serial.save("norb_prepro_global.pkl", standardize)
