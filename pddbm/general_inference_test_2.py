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
obs['H_hat'] = T.clip(obs['H_hat'],1e-7,1.-1e-7)
obs['G_hat'] = tuple([ T.clip(elem,1e-7,1.-1e-7) for elem in obs['G_hat']  ])

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

grad_G_sym = [ T.grad(obj, G_elem) for G_elem in G ]
grad_H_sym = T.grad(obj,H)
grad_S_sym = T.grad(obj,S)


grad_H = sharedX( H.get_value())
grad_S = sharedX( S.get_value())
grad_G = [ sharedX( G_elem.get_value())  for G_elem in G ]

updates = { grad_H : grad_H_sym, grad_S : grad_S_sym }

for grad_G_elem, grad_G_sym_elem in zip(grad_G,grad_G_sym):
    updates[grad_G_elem] = grad_G_sym_elem

compute_grad = function([V], updates = updates )

updates = { H : obs['H_hat'], S : obs['S_hat'] }

for G_elem, G_hat_elem in zip(G, obs['G_hat']):
    updates[G_elem] = G_hat_elem

init = function([V], trunc_kl,  updates = updates )

obj = function([V], obj)
print 'done'


alpha_list = [ .001, .005, .01, .05, .1 ]

def cache_values():

    global H_cache
    global S_cache
    global G_cache

    H_cache = H.get_value()
    S_cache = S.get_value()
    G_cache = [ G_elem.get_value() for G_elem in G ]


def clip(M):
    return np.clip(M,1e-7,1.-1e-7)

def goto_alpha(a):
    global H_cache
    global S_cache
    global G_cache

    #print 'a ',a

    assert not np.any(np.isnan(H_cache))
    piece_of_shit = grad_H.get_value()
    assert not np.any(np.isnan(piece_of_shit))
    if np.any(np.isinf(piece_of_shit)):
        print H.get_value()[np.isinf(piece_of_shit)]
        assert False
    mul = a * piece_of_shit

    assert not np.any(np.isnan(mul))

    diff = H_cache - mul

    assert not np.any(np.isnan(diff))

    fuck_you = clip( diff )

    assert not np.any(np.isnan(fuck_you))

    #print 'fuck you ',fuck_you.mean(),fuck_you.max(),fuck_you.min()

    H.set_value(fuck_you)
    #print 'H ',H.get_value().mean(), H.get_value().max(), H.get_value().min()
    S.set_value(S_cache-a*grad_S.get_value())
    for G_elem, G_cache_elem, grad_G_elem in zip(G, G_cache, grad_G):
        G_elem.set_value(clip(G_cache_elem-a*grad_G_elem.get_value()))

def norm_sq(s):
    return np.square(s.get_value()).sum()

def scale(s, a):
    s.set_value(s.get_value() * a)

def normalize_grad():
    n = sum( [ norm_sq(elem) for elem in grad_G ] )
    n += norm_sq(grad_H)
    n += norm_sq(grad_S)

    n = np.sqrt(n)

    for elem in grad_G:
        scale(elem, 1./n)
    scale(grad_H, 1./n)
    scale(grad_S, 1./n)

for i in xrange(num_batches):
    X = dataset.get_batch_design(batch_size)
    orig_kl = init(X)

    H.set_value(clip(H.get_value()))

    print 'batch ',i
    print orig_kl

    orig_H = H.get_value()
    orig_S = S.get_value()
    orig_G = [ G_elem.get_value() for G_elem in G ]

    for j in xrange(ga_updates):
        best_kl, best_alpha, best_alpha_ind = obj(X), 0., -1

        cache_values()
        compute_grad(X)
        normalize_grad()

        for ind, alpha in enumerate(alpha_list):
            goto_alpha(alpha)
            kl = obj(X)
            print '\t',alpha,kl

            if kl < best_kl:
                best_kl = kl
                best_alpha = alpha
                best_alpha_ind = ind

        print best_kl
        assert not np.isnan(best_kl)
        goto_alpha(best_alpha)

        if best_alpha_ind < 1 and alpha_list[0] > 3e-7:
            alpha_list = [ alpha / 3. for alpha in alpha_list ]
        elif best_alpha_ind > len(alpha_list) -2:
            alpha_list = [ alpha * 2. for alpha in alpha_list ]
        elif best_alpha_ind == -1 and alpha_list[0] <= 3e-7:
            break

    H_dist = np.sqrt( np.square( H.get_value() - orig_H ).sum() )
    S_dist = np.sqrt( np.square( S.get_value() - orig_S ).sum() )
    G_dist = np.sqrt( sum( [ np.square( G_elem.get_value() - G_orig_elem ).sum() for G_elem, G_orig_elem in zip(G, orig_G) ] ) )

    print 'kl improved by ',orig_kl-best_kl

    print 'H moved ',H_dist
    print 'S moved ',S_dist
    print 'G moved ',G_dist

    H_diffs = np.abs(H.get_value() - orig_H)
    S_diffs = np.abs(S.get_value() - orig_S)
    G_diffs = [ np.abs( G_elem.get_value() - orig_G_elem) for G_elem, orig_G_elem in zip(G, orig_G) ]

    print 'H diffs ',(H_diffs.min(), H_diffs.mean(), H_diffs.max() )
    print 'S diffs ',(S_diffs.min(), S_diffs.mean(), S_diffs.max() )
    print 'G diffs ',( min([G_diff_elem.min() for G_diff_elem in G_diffs]), sum([G_diff_elem.mean() for G_diff_elem in G_diffs])/float(len(G)),
            max([G_diff_elem.max() for G_diff_elem in G_diffs]) )
