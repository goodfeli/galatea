import numpy as np
rng = np.random.RandomState([2012, 11, 8])
from pylearn2.expr.nnet import inverse_sigmoid_numpy
from pylearn2.utils.serial import mkdir
import sys
import yaml
_, out_dir = sys.argv

num_jobs = 50

f = open('inpaint_job_template.yaml')
template = f.read()
f.close()

def uniform_between(a,b):
    return rng.uniform(np.minimum(a,b),np.maximum(a,b),(num_jobs,))

params = {}
names_before = []
names_before = locals().keys()


# Sparsity
use_sparsity_2 = uniform_between(0.,1.) > 0.5
layer_1_target = uniform_between(.01, .2)
layer_2_target = uniform_between(.05, .15)
layer_1_eps = (uniform_between(0., 1.) > .5) * rng.uniform(0., np.minimum(layer_1_target, .1))
layer_2_eps = (uniform_between(0., 1.) > .5) * rng.uniform(0., np.minimum(layer_2_target, .1))
layer_1_coeff = 10 ** uniform_between(-3., -1.)
layer_2_coeff = 10 ** uniform_between(-9., -2.)
layer_2_coeff *= use_sparsity_2

# Layer 1
layer_1_sparse_init = rng.randint(10,20, (num_jobs,))

switch = uniform_between(0., 1.) > 0.5
if_sparsity = switch * inverse_sigmoid_numpy(layer_1_target)
layer_1_init_bias = if_sparsity

# Layer 2
layer_2_dim = rng.randint(500,1500, (num_jobs))
layer_2_sparse_init = rng.randint(10,20, (num_jobs,))

switch = uniform_between(0., 1.) > 0.5
if_no_sparsity = switch * uniform_between(-2., 0.)
if_sparsity = switch * inverse_sigmoid_numpy(layer_2_target)

layer_2_init_bias = use_sparsity_2 * if_sparsity + (1-use_sparsity_2) * if_no_sparsity

# Class layer
class_sparse_init = rng.randint(10,20, (num_jobs,))

# Optimizer
reset_conjugate =  (uniform_between(0., 1.) > 0.5 )
max_iter = rng.randint(2, 5, (num_jobs,))

# Weight decay
wd1 = .0000005 * ( 10 ** uniform_between(-1., 1.))
wd2 = .0000005 * (10 ** uniform_between(-1., 1.) )
wdc = .0000005 * (10 ** uniform_between(-1., 1.) )

del switch
del if_no_sparsity
del if_sparsity

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


ref = {"layer_2_target":0.0890535860395, "layer_2_irange":0.0301747773266, "layer_2_init_bias":-0.741101442887, "layer_1_init_bias":-0.397164399345, "balance":0}
yaml.dump(ref)

mkdir(out_dir)
for i in xrange(num_jobs):
    cur_dir = out_dir +'/'+str(i)
    mkdir(cur_dir)
    path = cur_dir + '/stage_00_inpaint_params.yaml'

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

    path = cur_dir + '/stage_00_inpaint.yaml'
    f = open(path, 'w')
    f.write(template % obj)
    f.close()

