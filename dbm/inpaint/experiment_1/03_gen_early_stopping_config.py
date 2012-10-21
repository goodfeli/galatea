import numpy as np
rng = np.random.RandomState([2012, 10, 19])
from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.utils.serial import mkdir
import sys
import yaml
_, out_dir = sys.argv

num_jobs = 100

f = open('early_stopping_job_template.yaml')
template = f.read()
f.close()

def uniform_between(a,b):
    return rng.uniform(np.minimum(a,b),np.maximum(a,b),(num_jobs,))


params = {}
names_before = []
names_before = locals().keys()

# Batch size
switch = uniform_between(0., 1.) > 0.5
batch_size = switch * 2500 + (1-switch)*5000

# Optimizer
reset_alpha =  uniform_between(0., 1.) > 0.5
hacky_conjugacy = uniform_between(0., 1.) > 0.5
reset_conjugate = hacky_conjugacy * (uniform_between(0., 1.) > 0.5 )
updates_per_batch = rng.randint(1, 11, (num_jobs,))

params.update(locals())

for name in names_before:
    del params[name]


for key in sorted(params.keys()):
    val = params[key]
    if str(val.dtype) == 'bool':
        val = val.astype('int')
        params[key] = val
    assert val.shape == (num_jobs, )
    #print key,':',(val.min(),val.mean(),val.max())

for i in xrange(num_jobs):
    cur_dir = out_dir +'/'+str(i)
    path = cur_dir + '/stage_10_early_stop_params.yaml'

    obj = dict([(key, params[key][i]) for key in params])

    assert all([isinstance(key, str) for key in obj])
    assert all([isinstance(val, (int, float)) for val in obj.values()])

    # numpy has actually given us subclassed ints/floats that yaml doesn't know how to serialize
    for key in obj:
        if isinstance(obj[key], float):
            obj[key] = float(obj[key])
        elif isinstance(obj[key], int):
            obj[key] = int(obj[key])
        else:
            assert False

    output =  yaml.dump(obj, default_flow_style = False)

    f = open(path, 'w')
    f.write(output)
    f.close()

    path = cur_dir + '/stage_10_early_stop.yaml'
    f = open(path, 'w')
    f.write(template % obj)
    f.close()

