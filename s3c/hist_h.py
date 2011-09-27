import sys
import theano.tensor as T
from pylearn2.utils import serial

model_path = sys.argv[1]

shuffle = False
if len(sys.argv) > 2:
    assert sys.argv[2] == '--shuffle'
    #shuffle option gets random examples by skipping model.nhid ahead at the start
    #this way if using a random patches model we don't see the random patches that the
    #model uses as weights
    shuffle = True

model = serial.load(model_path)

model.make_Bwp()

stl10 = model.dataset_yaml_src.find('stl10') != -1

if not stl10:
    raise NotImplementedError("Doesn't support CIFAR10 yet")

if stl10:
    dataset = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_patches/data.pkl")

if shuffle:
    dataset.get_batch_design(model.nhid)

V_var = T.matrix()

mean_field = model.e_step.mean_field(V = V_var)

feature_type = 'exp_h'

if feature_type == 'exp_h':
    outputs = mean_field['H']
else:
    raise NotImplementedError()

from theano import function
f = function([V_var], outputs= outputs)

import matplotlib.pyplot as plt

while True:
    V = dataset.get_batch_design(1)
    y = f(V)

    print y.shape
    assert y.shape[0] == 1
    y = y[0,:]

    plt.hist(y, bins = 1000)
    plt.show()

    print 'Waiting...'
    x = raw_input()
    if x == 'q':
        break
    print 'running...'
