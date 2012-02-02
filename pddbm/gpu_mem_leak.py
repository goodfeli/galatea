#script to demonstrate that theano leaks memory on the gpu

import numpy as np
from pylearn2.utils import sharedX
from theano import function
import theano
import gc

s = [1,2]
before =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
W = sharedX(np.zeros((s[0],s[1])))
gc.collect()
gc.collect()
gc.collect()
after =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
diff = before[0] - after[0]
expected_diff = s[0]*s[1]*4

if diff > expected_diff:
    print "W uses ",str(float(diff)/float(expected_diff))," times more memory than needed."
    print "(",str(float(diff-expected_diff)/(1024. ** 2))," megabytes)"

grad  =sharedX(np.zeros(W.get_value().shape))
gc.collect()
gc.collect()
gc.collect()
after_after = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
diff = after_after[0] - after[0]

if diff > expected_diff:
    print "grad uses ",str(float(diff)/float(expected_diff))," times more memory than needed."


updates = { grad : W}

f = function([], updates = updates)

before =  theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
f()
gc.collect(); gc.collect(); gc.collect()
after = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
assert after[0] >= before[0]

