import numpy as np

class pow10(object):

    def __init__(self, offset = 0.):
        self.offset = offset

    def inverse(self, x):
        return np.log(x - self.offset) / np.log(10.)

    def __call__(self, x):
        return self.offset + 10. ** x
