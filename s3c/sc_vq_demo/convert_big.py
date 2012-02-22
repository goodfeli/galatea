import sys
import h5py
in_path, out_path = sys.argv[1:]
import numpy as np
print 'loading'
X = np.load(in_path)
print 'reshaping'
X = X.reshape(X.shape[0],X.shape[1] * X.shape[2] * X.shape[3])
assert len(X.shape) == 2
print 'saving'
f = h5py.File(out_path)
dst = f.create_dataset('X', data = X)
f.close()


