# Pseudo-code for hierarchical clustering

# Hierarchical clustering
# 
# Inputs
# dataset: n x d matrix with n examples with d features
# step   : number of recursion steps to perform
# k      : number of clusters at each step
#
# Output
#        : n x (k**1 + k**2 + ... + k**step) matrix representing the
#          the proximity of an example to each of the clusters centroids
def hc(dataset, step=5, k=2):

    def helper(dataset, step, k):
        means = kmeans(dataset, k)
        p = partition(dataset, means)
        vars = variances(dataset, p, means)
        px_given_k = probs(dataset, p, means, vars)
        
        # Recursion
        if step == 1:
            return (px_given_k, array([[] for i in range(dataset.shape[0])]))
        else:
            # Recursion step
            rec = [ helper(dataset[i == k,:], step-1, k) for i in range(k) ] 

            # Put each probability for a given step together
            lvl0 = [ rec[k][0] for k in range(k) ]
            lvl1 = [ rec[k][1] for k in range(k) ]

            return (px_given_k, hstack((hstack(lvl0), hstack(lvl1)))

    return hstack(helper(dataset, step, k))

# K-mean                
#
# Inputs:
# dataset: n x d matrix with n examples with d features
# k      : integer      number of clusters 
#
# Ouput:
#        :k x d matrix that contains the centroids of each cluster
def kmeans(dataset, k):
    pass

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
    pass

# Compute the variances for each of the cluster
# 
# Inputs:
# dataset: n x d matrix with n examples with d features
# p      : n x 1 partition matrix 
# means  : k x d matrix that contains the centroids of each cluster
# 
# Output:
#        : k-tuple of (d x d) covariance matrices for each cluster
def variances(dataset, p, means):
    pass

# Compute the probability that an example belongs to a given cluster
#
# Inputs:
# dataset: n x d matrix with n examples with d features
# p      : n x 1 partition matrix 
# means  : k x d matrix that contains the centroids of each cluster
# vars   : k-tuple of (d x d) covariance matrices of each cluster 
# 
# Output:
#        : n x k matrix that contains the probability that an example
#          belongs to a given cluster
def probs(dataset, p, means, vars):
    pass

        

