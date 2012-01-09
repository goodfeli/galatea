#arg1: model to evaluate
#arg2: batch size
#arg3: # batches
#arg4: # gradient ascent updates
#arg5: learning rate

import sys
import numpy as np
from pylearn2.utils import sharedX

model_path = sys.argv[1]

from pylearn2.utils import serial

print 'loading model...'
model = serial.load(model_path)
print 'done'

batch_size = int(sys.argv[2])
num_batches = int(sys.argv[3])
ga_updates = int(sys.argv[4])
learning_rate = float(sys.argv[5])

print 'defining em functional...'
import theano.tensor as T

V = T.matrix("V")
model.make_pseudoparams()
obs = model.e_step.variational_inference(V)

from pylearn2.models.s3c import S3C

needed_stats = S3C.expected_log_prob_vhs_needed_stats()

from pylearn2.models.s3c import SufficientStatistics

stats = SufficientStatistics.from_observations( needed_stats = needed_stats, V = V, ** obs )
em_functional = model.em_functional( stats = stats, H_hat = obs['H_hat'], S_hat = obs['S_hat'], var_s0_hat = obs['var_s0_hat'], var_s1_hat = obs['var_s1_hat'])
assert len(em_functional.type.broadcastable) == 0

print 'compiling function...'
from theano import function

H = sharedX(np.zeros((batch_size, model.nhid), dtype='float32'))
S = sharedX(np.zeros((batch_size, model.nhid), dtype='float32'))


new_stats = SufficientStatistics.from_observations( needed_stats = needed_stats, V = V, H_hat = H, S_hat = S,
            var_s0_hat = obs['var_s0_hat'], var_s1_hat = obs['var_s1_hat'])

obj = model.em_functional(stats = new_stats, H_hat = H, S_hat = S, var_s0_hat = obs['var_s0_hat'], var_s1_hat = obs['var_s1_hat'])

grad_H = T.grad(obj,H)
grad_S = T.grad(obj,S)

update = function([V], obj, updates =
                    { H : T.clip(H + learning_rate * grad_H, 1e-7, 1.-1e-7)
                        #, S : S + learning_rate * grad_S
                        } )

init = function([V], em_functional, updates = { H : obs['H_hat'], S : obs['S_hat'] })
print 'done'

print 'loading dataset...'
dataset = serial.load('${CIFAR10_PATCHES_6x6}')
print 'done'

from pylearn2.optimization.linear_cg import linear_cg



def find_careduce(var):
    if var.owner is None:
        return False
    op = var.owner.op
    opname = op.__class__.__name__

    if opname == 'CAReduce':
        print var.owner.inputs
        print var
        return True
    else:
        for ipt in var.owner.inputs:
            if find_careduce(ipt):
                print var
                return True



cg_update = function([V], obj,  updates = { S : linear_cg( fn = - obj, params = S, max_iters = 3) } )

for i in xrange(num_batches):
    X = dataset.get_batch_design(batch_size)
    em = init(X)

    print 'batch ',i
    print em

    for j in xrange(ga_updates):
        print 'ga: ',update(X)
        print 'cg: ',cg_update(X)
