import numpy as np
rng = np.random.RandomState([2012, 11, 8])
from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.utils.serial import mkdir
import sys
import yaml
_, out_dir = sys.argv

num_jobs = 100

f = open('validate_template.yaml')
template = f.read()
f.close()

for i in xrange(num_jobs):
    cur_dir = out_dir +'/'+str(i)

    path = cur_dir + '/stage_01_validate.yaml'
    f = open(path, 'w')
    f.write(template % { 'job' : i})
    f.close()

