import sys
import numpy as np

in_files = sys.argv[1:-1]
out_file = sys.argv[-1]

def num_feat(X):
    factors = X.shape[1:]

    rval = 1

    for factor in factors:
        rval *= factor

    return rval

print 'loading...'
in_X = [ np.load(in_file) for in_file in in_files ]
print 'reshaping...'
in_X = [ X.reshape(X.shape[0], num_feat(X) ) for X in in_X ]

print 'concatenating...'
out_X = np.concatenate(in_X, axis=1)

del in_X

print 'saving...'
np.save(out_file, out_X)
