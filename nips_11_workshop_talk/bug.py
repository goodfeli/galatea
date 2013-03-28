import numpy as np
from sklearn.decomposition import sparse_encode
HS = sparse_encode( np.random.randn(108,1600), np.random.randn(108,5000), alpha = 1./5000., algorithm='lasso_lars').T

