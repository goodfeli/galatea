from pylearn2.datasets.norb_small import FoveatedNORB

dataset = FoveatedNORB(which_set='train')

from pylearn2.datasets.preprocessing import Standardize

standardize = Standardize()

standardize.apply(dataset, can_fit=True)

from pylearn2.utils import serial
serial.save("norb_prepro.pkl", standardize)
