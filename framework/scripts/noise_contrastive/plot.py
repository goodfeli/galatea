import sys
from framework.utils import serial
model = serial.load(sys.argv[1])
model.redo_theano()

from framework.config import yaml_parse
import numpy as N

o = model.noise_var.get_value()

import matplotlib.pyplot as plt

d = yaml_parse.load(model.dataset_yaml_src)

while True:
    x = d.get_batch_design(1)

    if model.different_examples:
        x = [ x , d.get_batch_design(1) ]


    e = []
    l = []

    for i in xrange(1,100):
        cl = o * float(i) / 10.
        print cl
        model.noise_var.set_value(cl)

        l.append(cl)
        e.append(model.E_c_func(*x))

    plt.plot(N.asarray(l),N.asarray(e))

    plt.show()

    print 'waiting'
    x = raw_input()
    print 'running'
