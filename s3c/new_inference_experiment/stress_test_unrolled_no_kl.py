import sys

ignore, N  = sys.argv
N = int(N)

dataset_desc_path = 'cifar100_patches.yaml'

from pylearn2.config import yaml_parse
dataset = yaml_parse.load_path(dataset_desc_path)


from pylearn2.models.s3c import S3C, E_Step, Grad_M_Step, E_Step_Scan

NUM_EXAMPLES = 100
OVERKILL = 50

X = dataset.get_batch_design(NUM_EXAMPLES)

D = X.shape[1]

import numpy as np
rng = np.random.RandomState([7,12,43])
init_bias_hid = rng.uniform(-5.,0.,(N,))
init_mu = rng.uniform(-5.,5.,(N,))
init_alpha = rng.uniform(.1,15.,(N,))
init_B = rng.uniform(3.5,15.,(D,))

from pylearn2.utils.call_check import checked_call

def package_call(to_call, ** kwargs):
    return checked_call(to_call, kwargs)

model = package_call( S3C, nvis = D, nhid = N, local_rf_src = dataset,
            local_rf_shape = [6,6],
            local_rf_max_shape = dataset.view_shape()[0:2],
            local_rf_draw_patches = True,
            tied_B = False,
            init_bias_hid = init_bias_hid,
            init_mu = init_mu,
            init_alpha = init_alpha,
            init_B = init_B,
            irange = None,
            min_B = .01,
            max_B = 1e6,
            min_alpha = .01,
            max_alpha = 1e6,
            e_step = None,
            m_step = Grad_M_Step())
model.make_pseudoparams()


#this initial E step is just to help out the
#BatchGradientInference object
model.e_step = E_Step_Scan(
                clip_reflections = True,
                rho = 0.5,
                h_new_coeff_schedule = [.1 ] * OVERKILL,
                s_new_coeff_schedule = [.3 ] * OVERKILL)
model.e_step.register_model(model)

import theano.tensor as T
from theano import function

def get_needed_steps(ip, X):

    V = T.matrix()

    history = ip.infer(V, return_history = True)


    print 'compiling'
    t1 = time.time()
    f = function([V], history[-1].values() )
    t2 = time.time()

    print 'took ',t2-t1

    f(X)
    f(X)
    f(X)


    print 'running'
    t1 = time.time()
    kls = f(X)
    t2 = time.time()
    print 'runtime ',t2-t1





h_config_list =  [ .5 ]

TOL = .05

def make_ip(alg, s_config, h_config, n):

    if alg == 'scan':
        t = E_Step_Scan
    elif alg == 'non-scan':
        t = E_Step
    else:
        raise NotImplementedError()
    rval =  t(
                clip_reflections = True,
                rho = 0.5,
                h_new_coeff_schedule = [ h_config ] * n,
                s_new_coeff_schedule = [ s_config ] * n,
                )

    global model
    rval.register_model(model)
    return rval


import time

def time_run( ip, X):

    V = T.matrix()

    obs = ip.infer(V)

    H, S = obs['H_hat'], obs['S_hat']

    print 'compiling'
    t1 = time.time()
    f = function([V],[H,S])
    t2 = time.time()
    print 'took ',t2-t1

    print 'warming up'
    f(X)
    f(X)
    f(X)

    print 'running'
    t1 = time.time()
    H, S = f(X)
    t2 = time.time()

    return t2 - t1


s_config_list =  [ .5, .6, .7 ]

s_config = .7
h_config = .5
alg = 'non-scan'

ip = make_ip( alg, s_config, h_config, OVERKILL)

print 'running overkill'
n = get_needed_steps(ip, X)

if n < 0:
    print "FAILURE"
else:
    print n, ' STEPS REQUIRED'

