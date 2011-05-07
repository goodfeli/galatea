from framework.utils import serial
from framework.datasets import cifar10

class Streamer:
    def __init__(self, which_set):
        self.underlying = cifar10.CIFAR10(which_set = which_set)
        self.preprocessor = serial.load('/u/goodfeli/framework/recons_srbm/cifar10_preprocessor_2M.pkl')

    def get_batch_design(self, batch_size):
        self.preprocessor.
