from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np

class ZCA_Dataset(DenseDesignMatrix):

    def __init__(self,
            preprocessed_dataset,
            preprocessor):

        self.preprocessed_dataset = preprocessed_dataset
        self.preprocessor = preprocessor
        self.rng = self.preprocessed_dataset.rng

        self.X = preprocessed_dataset.X
        self.view_converter = preprocessed_dataset.view_converter
        self.y = None

        self.mn = self.X.min()
        self.mx = self.X.max()

        print 'inverting...'
        preprocessor.invert()
        print '...done inverting'

    def adjust_for_viewer(self, X):

        #rval = X - self.mn
        #rval /= (self.mx-self.mn)

        #rval *= 2.
        #rval -= 1.
        rval = X.copy()

        #rval = np.clip(rval,-1.,1.)


        for i in xrange(rval.shape[0]):
            rval[i,:] /= np.abs(rval[i,:]).max() + 1e-12

        return rval

    def adjust_to_be_viewed_with(self, X, other, per_example = False):

        #rval = X - self.mn
        #rval /= (self.mx-self.mn)

        #rval *= 2.
        #rval -= 1.


        rval = X.copy()

        if per_example:
            for i in xrange(rval.shape[0]):
                rval[i,:] /= np.abs(other[i,:]).max()
        else:
            rval /= np.abs(other).max()

        rval = np.clip(rval,-1.,1.)

        return rval

    def mapback_for_viewer(self, X):

        assert X.ndim == 2
        rval = self.preprocessor.inverse(X)
        rval = self.preprocessed_dataset.adjust_for_viewer(rval)

        return rval

    def mapback(self, X):
        return self.preprocessor.inverse(X)

