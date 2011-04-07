from pylearn.datasets import utlc
import numpy as N
from scipy import io
import theano.tensor as T
from theano import function
from scipy import sparse

pca_components = 512

devel, valid, test = utlc.load_sparse_dataset('terry')


#Constrain only to features that appear in intersection of devel, valid, test
def get_mask(mat):
    return N.asarray(mat.sum(axis=0) != 0)[0,:]
    #the N.asarray is needed because the CSR matrix sum method returns some
    #retarded kind of matrix object that just silently ignores the [] operator
    #reason #5,287 that I hate programming in python....


masks = [ get_mask(x) for x in [devel, valid, test] ]

mask = N.ones((devel.shape[1],))

for x in masks:
    mask *= x

print str(mask.sum())+' features remain'
idxs = N.nonzero(mask)[0]

devel = devel[:,idxs]
valid = valid[:,idxs]
test = test[:,idxs]



#Stack matrices and do PCA

X = sparse.vstack((devel, valid, test))
X = sparse.csr_matrix(X)


print X.shape

from framework.pca import SparseMatPCA

pca = SparseMatPCA(num_components = pca_components, minibatch_size = 10)

pca.train(X)


arg = pca.get_input_type()()

transformed = pca(arg)

f = function([arg],transformed)

devel, valid, test = [ f(x) for x in [devel,valid,test] ]


io.savemat('../data/terry_pca_%d.mat' % pca_components,{ 'devel' : devel, 'valid' : valid, 'test' : test } )



