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

from pylearn2.models.s3c import S3C

needed_stats = S3C.expected_log_prob_vhs_needed_stats()

from pylearn2.models.s3c import SufficientStatistics


params = []

for i in xrange(len(model.e_step.h_new_coeff_schedule)):
    param = sharedX(model.e_step.h_new_coeff_schedule[i], name='h'+str(i))
    model.e_step.h_new_coeff_schedule[i] = param
    params.append(param)

for i in xrange(len(model.e_step.s_new_coeff_schedule)):
    param = sharedX(model.e_step.s_new_coeff_schedule[i], name='s'+str(i))
    model.e_step.s_new_coeff_schedule[i] = param
    params.append(param)

param = sharedX(model.e_step.rho, name='rho')
model.e_step.rho = param
#params.append(param)

obs = model.e_step.variational_inference(V)
stats = SufficientStatistics.from_observations( needed_stats = needed_stats, V = V, ** obs )
obj = model.em_functional( stats = stats, H_hat = obs['H_hat'], S_hat = obs['S_hat'], var_s0_hat = obs['var_s0_hat'], var_s1_hat = obs['var_s1_hat'])

grads = T.grad(obj, params)

updates = {}

for param, grad in zip(params, grads):
    updates[param] = T.clip(param + learning_rate * grad, 1e-7, 1.-1e-7)

print 'compiling function...'
from theano import function


update = function([V], obj, updates = updates )
print 'done'

print 'loading dataset...'
dataset = serial.load('${CIFAR10_PATCHES_6x6}')
print 'done'

for i in xrange(num_batches):
    X = dataset.get_batch_design(batch_size)

    print 'batch ',i

    for j in xrange(ga_updates):
        print update(X)

        for param in params:
            print '\t'+param.name+' '+str(param.get_value())

            assert not np.any(np.isnan(param.get_value()))

