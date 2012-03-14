#arg1: model to evaluate
#arg2: batch size
#arg3: # batches
#arg4: # gradient ascent updates
#arg5: learning rate

import sys
import numpy as np
from pylearn2.utils import sharedX
from theano import config

model_path = sys.argv[1]

from pylearn2.utils import serial
from pylearn2.config import yaml_parse

print 'loading model...'
model = serial.load(model_path)
print 'done'

if len(sys.argv) > 2:
    batch_size = int(sys.argv[2])
else:
    batch_size = 100
if len(sys.argv) > 3:
    num_batches = int(sys.argv[3])
else:
    num_batches = 1
if len(sys.argv) > 4:
    ga_updates = int(sys.argv[4])
else:
    ga_updates = 100000
if len(sys.argv) > 5:
    learning_rate = float(sys.argv[5])
else:
    learning_rate = .001

print 'loading dataset...'
dataset = yaml_parse.load(model.dataset_yaml_src)
print 'done'

print 'defining em functional...'
import theano.tensor as T

V = T.matrix("V")
#draw this batch regardless of whether testing is on so that the
#loop below always sees the same data
X = dataset.get_batch_design(batch_size)
if config.compute_test_value != 'off':
    model.s3c.test_batch_size = batch_size
    V.tag.test_value = X

model.make_pseudoparams()
obs = model.inference_procedure.infer(V)

from pylearn2.models.s3c import S3C

needed_stats = S3C.expected_log_prob_vhs_needed_stats()

from pylearn2.models.s3c import SufficientStatistics

#stats = SufficientStatistics.from_observations( needed_stats = needed_stats, V = V, ** obs )
#em_functional = model.em_functional( stats = stats, H_hat = obs['H_hat'], S_hat = obs['S_hat'], var_s0_hat = obs['var_s0_hat'], var_s1_hat = obs['var_s1_hat'])

trunc_kl = model.inference_procedure.truncated_KL(V, obs)

if config.compute_test_value != 'off':
    assert not np.any(np.isnan(trunc_kl.tag.test_value))

assert len(trunc_kl.type.broadcastable) == 0

print 'compiling function...'
from theano import function

G = [ sharedX(np.zeros((batch_size, rbm.nhid), dtype='float32')) for rbm in model.dbm.rbms ]
H = sharedX(np.zeros((batch_size, model.s3c.nhid), dtype='float32'))
S = sharedX(np.zeros((batch_size, model.s3c.nhid), dtype='float32'))

new_stats = SufficientStatistics.from_observations( needed_stats = needed_stats, V = V, H_hat = H, S_hat = S,
                    var_s0_hat = obs['var_s0_hat'], var_s1_hat = obs['var_s1_hat'])


obj = model.inference_procedure.truncated_KL( V, {
    "H_hat" : H,
    "S_hat" : S,
    "var_s0_hat" : obs['var_s0_hat'],
    "var_s1_hat" : obs['var_s1_hat'],
    "G_hat" : G
    } )

grad_G = [ T.grad(obj, G_elem) for G_elem in G ]
grad_H = T.grad(obj,H)
grad_S = T.grad(obj,S)

learning_rate_sym = T.scalar()

updates = { H : T.clip(H - learning_rate_sym * grad_H, 0., 1.), S : S - learning_rate_sym * grad_S }

for G_elem, grad_G_elem in zip(G,grad_G):
    updates[G_elem] = T.clip(G_elem - learning_rate_sym * grad_G_elem, 0., 1.)

update = function([V, learning_rate_sym], obj, updates = updates )

updates = { H : obs['H_hat'], S : obs['S_hat'] }

for G_elem, G_hat_elem in zip(G, obs['G_hat']):
    updates[G_elem] = G_hat_elem

init = function([V], trunc_kl,  updates = updates )
print 'done'


for i in xrange(num_batches):
    X = dataset.get_batch_design(batch_size)
    prev_kl = init(X)

    print 'batch ',i
    print prev_kl

    for j in xrange(ga_updates):
        kl = update(X, learning_rate)
        print kl

        if kl > prev_kl:
            learning_rate *= .99
            print 'scaled learning rate to ',learning_rate
        prev_kl = kl

