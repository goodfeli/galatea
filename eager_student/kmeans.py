from scipy.cluster.vq import kmeans
import numpy as N

class KMeans:
    def __init__(self, k, verbose):
        self.k = k
        self.verbose = verbose

    def train(self, dataset):
        X = dataset.get_design_matrix()

        mu, err = kmeans(X, self.k)

        N.save( 'mu.npy', mu)
