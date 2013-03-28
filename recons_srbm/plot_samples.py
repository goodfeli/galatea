import numpy as N
import sys
from framework.utils import serial
import matplotlib.pyplot as plt
from framework.config import yaml_parse

model = serial.load(sys.argv[1])
model.redo_theano()

#X = N.random.RandomState([1,2,3]).randn(500,2)
dataset = yaml_parse.load(model.dataset_yaml_src)
X = dataset.get_batch_design(500)

while True:
    plt.scatter(X[:,0],X[:,1])
    plt.show()


    print 'waiting'

    x = raw_input()
    if x == 'q':
        break

    try:
        niter = int(x)
    except:
        niter = 1

    print 'running'

    for i in xrange(niter):
        X = model.sample(X)
