#script to demonstrate that common subexpressions slow down python loops
#while unrolled python loops optimize them out
import sys
sys.setrecursionlimit(50000)
import numpy as np
from theano import function
from pylearn2.utils import sharedX
from theano import tensor as T
import time
from theano import config

batch_size = 100
rep_size = 4000
num_rounds = 3

def run_timed_trial(f, num_warmups = 3, num_trials = 5):
    print 'Warming up...'
    for i in xrange(num_warmups):
        f()
    trials = []
    for i in xrange(num_trials):
        t1 = time.time()
        f()
        t2 = time.time()
        trials.append(t2-t1)
    arr = np.asarray(trials)
    print 'Mean time: ',np.mean(arr)
    print 'Standard deviation: ',np.std(arr)

rng = np.random.RandomState([1,2,3])

W = sharedX( rng.randn(rep_size,rep_size) , name = 'W')
V = sharedX( np.zeros((batch_size, rep_size) ) , name = 'V')
H = sharedX( np.zeros((batch_size, rep_size) ) , name = 'H')
G = sharedX( np.zeros((batch_size, rep_size) ) , name = 'G')

init_V = sharedX(rng.uniform(0,1,(batch_size,rep_size)))
init_H = sharedX(rng.uniform(0,1,(batch_size,rep_size)))

print 'Compiling init'
init = function([], updates = { V : init_V, H : init_H } )

def update_V(cur_H):
    return T.nnet.sigmoid(T.dot(cur_H,W.T))

def update_H(cur_V):
    return T.nnet.sigmoid(T.dot(cur_V,W) + T.dot(G, W.T) )


G_contrib = sharedX( np.dot( G.get_value(), W.get_value().T) )


print 'Compiling single updates'
do_one_V_update = function([], updates = { V : update_V(H) } )
do_one_H_update = function([], updates = { H : update_H(V) } )
do_one_optimized_H_update = function([], updates = { H : T.nnet.sigmoid(T.dot(V, W) + G_contrib) } )

def python_loop():
    init()
    for i in xrange(num_rounds):
        do_one_V_update()
        do_one_H_update()

def optimized_python_loop():
    init()
    for i in xrange(num_rounds):
        do_one_V_update()
        do_one_optimized_H_update()

cur_V = V
cur_H = H

for i in xrange(num_rounds):
    cur_V = update_V(cur_H)
    cur_H = update_H(cur_V)

print 'Compiling unrolled theano'
unrolled_theano = function([], updates = { V : cur_V, H : cur_H } )

from theano.printing import min_informative_str
print min_informative_str(unrolled_theano.maker.env.outputs[0])
assert False

def unrolled_loop():
    init()
    unrolled_theano()


print 'Timing python loop'
run_timed_trial(python_loop)

print 'Timing unrolled loop'
run_timed_trial(unrolled_loop)

print 'Timing optimized python loop'
run_timed_trial(optimized_python_loop)
