from framework import pca
from framework.distributions import multinomial
import numpy as N

class Preprocessor:
    def __init__(self, src, num_views, pca_dim, subset_dim):

        assert subset_dim <= pca_dim

        self.numviews, self.pca_dim, self.subset_dim = num_views, pca_dim, subset_dim

        self.pca = pca.CovEigPCA(num_components = pca_dim)

        print 'training pca'

        self.pca.train(src.get_design_matrix())

        print 'picking random views'

        projector = self.pca.W.get_value()

        subset_projectors = []

        rng = N.random.RandomState([1,2,3])

        for i in xrange(num_views):
            subset_components = []


            values = list( self.pca.v.get_value() )

            for j in xrange(subset_dim):
                idx = multinomial.Multinomial(rng, N.asarray(values), renormalize = True).sample_integer(1)[0]
                del values[idx]
                subset_components.append(projector[:,idx])

            subset_projectors.append(N.concatenate(subset_components,axis=1))

        #note-- src should already be standardized, so we don't need to do anything other than
        #projection here
        print 'computing views'

        views = [ N.dot( src.get_design_matrix(), N.dot(p,p.T)) for p in subset_projectors ]


        print 'computing preprocessed data'
        self.X = N.concatenate(views, axis=1)

        print 'done preprocessing'

    def get_design_matrix(self):
        return self.X
