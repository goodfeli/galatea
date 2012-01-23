#arg1: layer 1 model
#arg2: layer 2 model

import sys

l1, l2, idx, alpha = sys.argv[1:]
idx = int(idx)
alpha = float(alpha)

from pylearn2.utils import serial
l1 = serial.load(l1)
l2 = serial.load(l2)

from pylearn2.models.dbm import DBM

dbm =  DBM(negative_chains = 1,
           monitor_params = 0,
           rbms = [ l2 ])


from galatea.pddbm.pddbm import PDDBM, InferenceProcedure

pddbm = PDDBM(
        dbm = dbm,
        s3c = l1,
        inference_procedure = InferenceProcedure(
            schedule = [ ['s',1.],  ['h',1.],   ['g',0],   ['h', 0.4], ['s',0.4],
                         ['h',0.4], ['g',0],   ['h',0.4], ['s',0.4],  ['h',0.4],
                         ['g',0],   ['h',0.4], ['s',0.4], ['h', 0.4], ['g',0],
                         ['h',0.4], ['g',0],   ['h',0.4], ['s', 0.4], ['h',0.4] ],
            monitor_kl = 0,
            clip_reflections =  1,
            rho = 0.5)
        )

from pylearn2.utils import sharedX
import numpy as np
import theano.tensor as T

h = l2.weights.get_value()[:,idx]
v = np.dot(l1.W.get_value() * l1.mu.get_value(),h)

init = v / np.sqrt(np.square(v).sum())

X = sharedX( np.zeros((1,l1.nvis)) + init )

obs = pddbm.inference_procedure.infer(X)

g = obs['G_hat'][0][0,0]

grad = T.grad(g,X)


step_X = X + alpha * grad

norm =  T.sqrt(T.sqr(X).sum())

renormed_X = step_X #/ norm

from theano import function

f = function([],[g,norm],updates = { X : renormed_X })

while True:
    a,b =  f()

    print a,' ',b

    if a > .75:
        break

from pylearn2.utils import image
from pylearn2.config import yaml_parse

dataset = yaml_parse.load( l1.dataset_yaml_src )

X = dataset.get_topological_view( X.get_value())

X /= np.abs(X).max()

image.show(X[0,...])



