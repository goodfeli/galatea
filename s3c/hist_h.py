import sys
import theano.tensor as T
from pylearn2.utils import serial

model_path = sys.argv[1]


model = serial.load(model_path)

model.make_Bwp()

stl10 = model.dataset_yaml_src.find('stl10') != -1

if not stl10:
    raise NotImplementedError("Doesn't support CIFAR10 yet")

if stl10:
    dataset = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_patches/data.pkl")

V_var = T.matrix()

mean_field = model.e_step.mean_field(V = V_var)

feature_type = 'exp_h'

if feature_type == 'exp_h':
    outputs = [ mean_field['H'] ]
else:
    raise NotImplementedError()

from theano import function
f = function([V_var], outputs= outputs)

import matplotlib.pyplot as plt

while True:
    V = dataset.get_batch_design(1)
    y = f(V)

    plt.hist(y)
    plt.show()

    print 'Waiting...'
    x = raw_input()
    if x == 'q':
        break
    print 'running...'
