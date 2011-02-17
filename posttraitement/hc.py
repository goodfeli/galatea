from numpy import *
from scipy.cluster import vq

# Pseudo-code for hierarchical clustering

def kmnb(s,k):
    assert s >= 1
    nb = k
    while s > 1:
        nb = nb + k**s
        s -= 1
    return nb

def dist(m, v):
    return sum((m-v)**2, axis=1)

# Hierarchical clustering
# 
# Inputs
# dataset : n x d matrix with n examples with d features
# max_step: number of recursion steps to perform
# k       : number of clusters at each step
#
# Output
#         : 3-uple 
#          nb x d matrix of clusters centroids,
#          list of nb covariance matrices
#          array of nb sample numbers
#          with nb = (k**1 + k**2 + ... + k**step)
def hc(dataset, max_step=5, k=2):
    # Nb of clusters to find 
    nb = kmnb(max_step, k)
    # Cluster centroids
    means = zeros((nb, dataset.shape[1]))
    # Variance
    vars  = [ None for i in range(nb) ]
    # Nb of examples in a given cluster
    snb   = zeros((nb,))

    def helper(dataset, step, base_idx, offset):
        cs = kmeans(dataset, k)
        p = partition(dataset, cs)

        index = base_idx + k*offset
        means[index:index+k,:] = cs
        #print "Step %i ,Computing range [%i,%i["%(step,index, index+k)

        for i in range(k):
            snb[index+i] = sum(p == i)
            vars[index+i] = var(dataset[p == i], axis=0)

            if step < max_step:
                helper(dataset[p == i], step+1, base_idx + k**step, k*offset + i)

    helper(dataset, 1, 0, 0)
    return (means, vars, snb)

# K-mean                
#
# Inputs:
# dataset: n x d matrix with n examples with d features
# k      : integer      number of clusters 
#
# Ouput:
#        :k x d matrix that contains the centroids of each cluster
def kmeans(dataset, k):
    # TODO: Do we need whitening?
    return vq.kmeans(dataset, k)[0]

# Partition the data according to the nearest centroid
#    
# Inputs
# dataset: n x d matrix with n examples with d features
# means  : k x d matrix that contains the centroids of each cluster
#
# Output
#        : n x 1 matrix that contains the index to the nearest cluster 
#          to the nth matrix
def partition(dataset, means):
    n = dataset.shape[0]
    k = means.shape[0]
    p = zeros((n,k))

    for i in range(k):
        p[:,i] = dist(dataset, means[i,:]) 
    return argmin(p, axis=1)

# Compute the probability that an example belongs to a given cluster
#
# Inputs:
# dataset: n x d matrix with n examples with d features
# means  : k x d matrix that contains the centroids of each cluster
# vars   : k-tuple of (d x d) covariance matrices of each cluster 
#                  or (d x 1) variance vectors
# snb    : k x 1 matrix of sample numbers for each clusters 
# Output:
#        : n x k matrix that contains the probability that an example
#          belongs to a given cluster
def probs(dataset, means, vars):
    pass

if __name__ == "__main__":
    dataset = array([[-4, 2],\
                     [-3, 2],\
                     [-4, 1],\
                     [-3, 1],\
                     [-4,-1],\
                     [-3,-1],\
                     [-4,-2],\
                     [-3,-2],\
                     [ 3, 2],\
                     [ 4, 2],\
                     [ 3, 1],\
                     [ 4, 1],\
                     [ 3,-1],\
                     [ 4,-2],\
                     [ 3,-2],\
                     [ 4,-1]], dtype='float')

    (means, vars, snb) = hc(dataset, 2, 2)
    print means
    print vars
    print snb

        

