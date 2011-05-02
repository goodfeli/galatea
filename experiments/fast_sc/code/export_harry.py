from pylearn.datasets import utlc
import numpy as N
from scipy import io

devel, valid, test = utlc.load_ndarray_dataset('harry')

X = N.concatenate((devel,valid,test),axis=0)

print X.shape

io.savemat('../data/harry_all.mat',{ 'X': X })
