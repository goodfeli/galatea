import sys
in_path, out_path = sys.argv[1:]
from scipy import io
import numpy as np
print 'loading'
X = np.load(in_path)
if len(X.shape) != 2:
    print 'reshaping'
    X = X.reshape(X.shape[0],X.shape[1] * X.shape[2] * X.shape[3])
assert len(X.shape) == 2
print 'saving'
#io.savemat(out_path,{'X_chunk_0':X[0:25000,:], 'X_chunk_1':X[25000:,:]})

assert X.shape[0] in [50000,10000]

d = {}

for i in xrange(X.shape[0] / 5000):
    d[ 'X_chunk_' + str(i) ] = X[i*5000:(i+1)*5000,:]

io.savemat( out_path, d )

del X

print 'reloading to verify it worked'
test = io.loadmat(out_path)
