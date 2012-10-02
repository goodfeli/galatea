from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np

class ZCA_Dataset(DenseDesignMatrix):

    def __init__(self,
            preprocessed_dataset,
            preprocessor,
            convert_to_one_hot = True,
            start = None,
            stop = None):

        self.preprocessed_dataset = preprocessed_dataset
        self.preprocessor = preprocessor
        self.rng = self.preprocessed_dataset.rng

        if start is not None:
            self.X = preprocessed_dataset.X[start:stop,:]
            assert self.X.shape[0] == stop-start
        else:
            self.X = preprocessed_dataset.X
        self.view_converter = preprocessed_dataset.view_converter
        self.y = preprocessed_dataset.y

        if convert_to_one_hot:
            if not ( self.y.min() == 0):
                raise AssertionError("Expected y.min == 0 but y.min == "+str(self.y.min()))
            nclass = self.y.max() + 1
            y = np.zeros((self.y.shape[0], nclass), dtype='float32')
            for i in xrange(self.y.shape[0]):
                y[i,self.y[i]] = 1.
            self.y = y

        #self.mn = self.X.min()
        #self.mx = self.X.max()

        print 'inverting...'
        preprocessor.invert()
        print '...done inverting'

    def has_targets(self):
        return self.preprocessed_dataset.has_targets()

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

