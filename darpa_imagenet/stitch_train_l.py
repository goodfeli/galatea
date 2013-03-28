import numpy as np

pieces = []

for i in xrange(25):
    print i
    pieces.append(np.load('/mnt/scratch/S3C_5625/trainl_%d_S3C_5625.npy' % (i,) ) )

print 'concat'
X = np.concatenate(pieces, axis=0)
del pieces

import gc
gc.collect()

print 'save'
np.save('/mnt/scratch/stitched/trainl.npy', X)
