from theano import function
from pylearn2.utils import serial
from pylearn2.utils import sharedX

print 'Loading'
dataset = serial.load("/data/lisatmp/goodfeli/esp_bow.pkl")

X = sharedX(dataset.X)

model = serial.load("sparse_rbm.pkl")

print 'Compiling'
model.niter = 1
feat = model.mf(X)[-1][0]
f = function([], feat)

print 'Running'
feat = f()

print 'Output shape: ',feat.shape

print 'Saving'
import numpy as np
np.save('/data/lisatmp/goodfeli/esp/wordfeat.npy', feat)
