from theano import function
from theano.tensor.shared_randomstreams import RandomStreams

theano_rng = RandomStreams(137)

def debug():
    theano_rng.seed(5)
    return func()

def make_func():
    global func
    rval = theano_rng.normal(size = (1,), avg = 0., std = 1.)
    func = function([], rval)


make_func()
first = debug()
second = debug()
assert first == second

make_func()
third = debug()

assert first == third
