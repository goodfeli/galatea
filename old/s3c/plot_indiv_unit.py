#arg1: model to evaluate

import sys
import numpy as np
from pylearn2.utils import sharedX
from theano import function
import matplotlib.pyplot as plt
from pylearn2.config import yaml_parse

model_path = sys.argv[1]

from pylearn2.utils import serial

print 'loading model...'
model = serial.load(model_path)
print 'done'

print 'defining em functional...'
import theano.tensor as T

V = T.matrix("V")
model.make_pseudoparams()
hist = model.e_step.variational_inference(V, return_history = True)

outputs = []

for obs in hist:
    outputs.append(obs['H_hat'])
    outputs.append(obs['S_hat'])

f = function([V], outputs)


patches = yaml_parse.load(model.dataset_yaml_src)


while True:
    i = input('Example: ')

    raw = f(patches.X[i:i+1,:])


    print (raw[-2].min(), raw[-2].mean(), raw[-2].max())
    print 'Units above .5:', np.nonzero(raw[-2] > 0.5)[1]
    j = input('Unit: ')

    h = []
    s = []

    for k in xrange(0,len(raw),2):
        h.append(raw[k][0,j])
        s.append(raw[k+1][0,j])

    plt.plot(h, label='h')
    plt.plot(s, label='s')

    plt.legend(bbox_to_anchor=(1.05, 1),  loc=2, borderaxespad=0.)

    plt.show()



