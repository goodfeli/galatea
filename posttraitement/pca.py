import numpy
from scipy import linalg

def pca(X):
    (U, s, Vt) = linalg.svd(X)
    return numpy.dot(U, linalg.diagsvd(s, X.shape[0], X.shape[1]))
