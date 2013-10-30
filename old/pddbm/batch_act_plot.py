import sys

model_path = sys.argv[1]

from pylearn2.utils import serial

model = serial.load(model_path)

assert model.s3c is model.inference_procedure.s3c_e_step.model

model.make_pseudoparams()

from pylearn2.config import yaml_parse

dataset = yaml_parse.load(model.dataset_yaml_src)

batch_size = 100
num_batches = 5
m = batch_size * num_batches
nh = model.s3c.nhid

import numpy as np
H = np.zeros((m,nh))

import theano.tensor as T
from theano import function

X = T.matrix('X')

features = model.inference_procedure.infer(X)['H_hat']
features.name = 'features'

f = function([X],features)

for i in xrange(num_batches):
    X = dataset.get_batch_design(batch_size)
    H[i*batch_size:(i+1)*batch_size,:] = f(X)


h = H.mean(axis=0)
proj = np.dot(H,h)

#sort examples by projection along the mean vector
ranking = sorted(zip(proj,range(proj.shape[0])))

new_H = H.copy()

for i, t in enumerate(ranking):
    new_H[i,:] = H[t[1],:]
H = new_H

#sort units by mean activation
ranking = sorted(zip(h,range(h.shape[0])))

new_H = H.copy()

for i, t in enumerate(ranking):
    new_H[:,i] = H[:,t[1]]
H = new_H



from pylearn2.utils import image

image.show(H)

