# Trying to figure out how much error is OK for probabilistic_max_pooling.py
import numpy as np

rng = np.random.RandomState([2012,9,19])

m = 10000

pool_rows = 2
pool_cols = 3
tot = pool_rows * pool_cols + 1

mx = 0
for j in xrange(5*16*10*3):
    z = rng.randn( tot )*2.-3.
    z[-1] = 0.
    mult = np.exp(z) / np.exp(z).sum()

    acc = np.zeros(z.shape)

    for i in xrange(m):
        sample = rng.multinomial(1, mult, size = 1)[0,:]
        acc += sample

    est = acc / float(m)

    err =  np.abs(mult-est).max()
    mx = max(err, mx)

    print err, mx
