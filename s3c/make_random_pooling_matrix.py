from scipy import sparse
import numpy as np
from pylearn2.utils import serial

num_out_features = 3*3*1600
num_filters = 1600
map_rows = 7 #this is based on using 4x4 superpixels
map_cols = 7

num_in_features = map_rows * map_cols * num_filters

def rect_to_vec(idx, top, left, bottom, right):
    mat = np.zeros((map_rows, map_cols, num_filters))
    mat[top:bottom+1,left:right+1]=1.
    mat /= mat.sum()

    rval = mat.reshape( (num_in_features,) )

    return rval

#make LIL matrix
lil = sparse.lil_matrix((num_out_features,num_in_features))

rng = np.random.RandomState([1,2,3])

def write_vec(row, vec):
    global lil
    ids ,= np.nonzero(vec)
    for idx in ids:
        lil[row,idx] = vec[idx]

for i in xrange(num_out_features):
    print i
    #choose random filter index
    idx = rng.randint(num_filters)
    #choose random rectangle
    top = rng.randint(map_rows)
    bottom = rng.randint(top,map_rows)
    left = rng.randint(map_cols)
    right = rng.randint(left,map_cols)
    #get vector for that rectangle
    vec = rect_to_vec(idx,top,left,bottom,right)
    #write that vector into LIL matrix
    write_vec(i,vec)

print 'converting to csr...'
csr = sparse.csr_matrix(lil)
del lil

print 'saving...'
serial.save('pooling_mat.pkl',csr)
