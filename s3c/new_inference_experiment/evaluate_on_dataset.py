import sys

ignore, dataset_desc_path, N_str = sys.argv

from pylearn2.config import yaml_parse
dataset = yaml_parse.load_path(dataset_desc_path)

N = int(N_str)

from pylearn2.models.s3c import S3C, Grad_M_Step
from galatea.s3c.s3c import E_Step_Scan, E_Step_CG_Scan

NUM_EXAMPLES = 100
OVERKILL = 200

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


#this initial E step is just to help out the
#BatchGradientInference object
model.e_step = E_Step_Scan(
                clip_reflections = True,
                rho = 0.5,
                h_new_coeff_schedule = [.1 ] * OVERKILL,
                s_new_coeff_schedule = [.1 ] * OVERKILL)
model.e_step.register_model(model)


from galatea.pddbm.batch_gradient_inference_theano import BatchGradientInference

heavy_inference = BatchGradientInference(model)

result = heavy_inference.run_inference(X)

target = result['kl']



s_config_dict = {
        'hack' : [ .1, .2, .3 ],
        'cg' : [ 1, 2, 3 ]
        }

TOL = .05

def make_ip(alg, s_config, h_config, n):

    if alg == 'hack':
        rval =  E_Step_Scan(
                clip_reflections = True,
                rho = 0.5,
                h_new_coeff_schedule = [ h_config ] * n,
                s_new_coeff_schedule = [ s_config ] * n,
                )
    elif alg == 'cg':
        rval = E_Step_CG_Scan(
                h_new_coeff_schedule = [ h_config ] * n,
                s_max_iters = [ s_config ] * n)
    else:
        assert False

    global model
    rval.register_model(model)
    return rval

import theano.tensor as T
from theano import function

def get_needed_steps(ip, X, target, tol):

    V = T.matrix()

    history = ip.infer(V, return_history = True)

    kls = [ ip.truncated_KL(V, obs = history_elem, Y = None).mean() for history_elem in history ]

    print 'compiling'
    f = function([V], kls )

    print 'running'
    kls = f(X)

    for i in xrange(len(kls)):
        if kls[i] < target + tol:
            return i - 1
    return -1

import time

def time_run( ip, X):

    V = T.matrix()

    obs = ip.infer(V)

    H, S = obs['H_hat'], obs['S_hat']

    print 'compiling'
    f = function([V],[H,S])


    print 'warming up'
    f(X)
    f(X)
    f(X)

    print 'running'
    t1 = time.time()
    H, S = f(X)
    t2 = time.time()

    return t2 - t1


for alg in ['hack','cg']:
    best_time = 1e30
    for s_config in s_config_dict[alg]:
        for h_config in [ .1, .2, .3 ]:
            print 'testing ',alg,' ',s_config,' ',h_config


            ip = make_ip( alg, s_config, h_config, OVERKILL)

            print 'running overkill'
            n = get_needed_steps(ip, X, target, TOL)

            if n < 0:
                print "FAILURE"
            else:
                print n, ' STEPS REQUIRED'

                ip = make_ip( alg, s_config, h_config, n)

                print 'doing timed run'
                t = time_run( ip, X )

                print 'time required: ',t

                if t < best_time:
                    best_time = t
            #end if n < 0
        #end h_config loop
    #end s_config loop
    print 'BEST TIME THIS ALGORITHM ('+alg+'): ',best_time
#end alg loop


