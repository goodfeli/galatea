'''TODO: module-level docstring.'''
__authors__ = "Jason Yosinski"
__license__ = "3-clause BSD"
__maintainer__ = "Jason Yosinski"
__email__ = "yosinski@cs.cornell"



######### Old imports
import functools

import warnings
import numpy as np
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    FiniteDatasetIteratorPyTables,
    resolve_iterator_class
)
import copy
import ipdb as pdb
# Don't import tables initially, since it might not be available
# everywhere.
tables = None


from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets import control
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
from pylearn2.utils import safe_zip
from theano import config
########## Old imports

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix



class TemporalDenseDesignMatrix(DenseDesignMatrix):
    '''
    A class for representing datasets that can be stored as a dense design
    matrix, but whose examples are slices of width >= 2 rows each.
    '''

    _default_seed = (17, 2, 946)

    def __init__(self, X=None, topo_view=None, y=None,
                 view_converter=None, axes = ('b', 0, 1, 2, 'c'),
                 rng=_default_seed, preprocessor = None, fit_preprocessor=False):
        '''
        TODO: rewrite or just inherit...
        same as DenseDesignMatrix...???
        
        Parameters
        ----------

        X : ndarray, 2-dimensional, optional
            Should be supplied if `topo_view` is not. A design
            matrix of shape (number examples, number features)
            that defines the dataset.
            XXXXXXXXXXX not allowed
        topo_view : ndarray, optional
            Should be supplied if X is not.  An array whose first
            dimension is of length number examples. The remaining
            dimensions are xamples with topological significance,
            e.g. for images the remaining axes are rows, columns,
            and channels.
            TODO: time is 0, ii is 1, jj is 2
        y : ndarray, 1-dimensional(?), optional
            Labels or targets for each example. The semantics here
            are not quite nailed down for this yet.
        view_converter : object, optional
            An object for converting between the design matrix
            stored internally and the data that will be returned
            by iterators.
        rng : object, optional
            A random number generator used for picking random
            indices into the design matrix when choosing minibatches.
        '''

        assert topo_view is not None, (
            'For TemporalDenseDesignMatrix, must provide topo_view (not X)'
        )

        assert axes == ('b', 0, 1, 2, 'c')

        reduced_axes = ('b', 0, 1, 'c')
        
        super(TemporalDenseDesignMatrix, self).__init__(
            X = X,
            topo_view = topo_view,
            y = y,
            view_converter = view_converter,
            axes = reduced_axes,
            rng = rng,
            preprocessor = preprocessor,
            fit_preprocessor = fit_preprocessor
        )

        self._X = self.X
        self.X = None   # prevent other access

    def set_topological_view(self, topo_view, axes=('b', 0, 1, 'c')):
        '''
        Sets the dataset to represent topo_view, where topo_view is a batch
        of topological views of examples.

        Parameters
        ----------
        topo_view : ndarray
            An array containing a design matrix representation of training
            examples.
        '''
        
        assert not np.any(np.isnan(topo_view))
        frames = topo_view.shape[axes.index('b')]    # pretend frames come in as batch dim
        rows = topo_view.shape[axes.index(0)]
        cols = topo_view.shape[axes.index(1)]
        channels = topo_view.shape[axes.index('c')]

        # leave out frames...
        self.view_converter = DefaultViewConverter([rows, cols, channels], axes=axes)
        
        self.X = self.view_converter.topo_view_to_design_mat(topo_view)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = self.view_converter.topo_space
        assert not np.any(np.isnan(self.X))

        # Update data specs
        X_space = VectorSpace(dim = frames * rows * cols * channels)
        X_source = 'features'

        assert self.y is None, 'y not supported now'
        space = X_space
        source = X_source

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (X_space, X_source)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        '''thin wrapper... TODO: doc'''

        assert mode == 'shuffled_sequential', (
            'Only shuffled_sequential mode is supported'
        )
        assert data_specs != None, 'Must provide data_specs'
        assert len(data_specs) == 2, 'data_specs must include only one tuple for "features"'
        assert type(data_specs[0]) is CompositeSpace, 'must be composite space...??'
        assert data_specs[0].num_components == 1, 'must only have one component, features'
        assert data_specs[1][0] == 'features', 'data_specs must include only one tuple for "features"'

        output_space = data_specs[0].components[0]
        num_frames = output_space.shape[0]

        if num_batches is None:
            num_batches = 10  # another hack... just determines how often new iterators will be created?
        base_num_batches = num_batches * batch_size




        # Iterates through ONE example at a time
        # BEGIN HUGE HACK  (enable self.X access just for this function)
        self.X = self._X
        base_iterator = super(TemporalDenseDesignMatrix, self).iterator(
            mode='random_slice',  # to return continguous bits
            batch_size=num_frames,
            num_batches=base_num_batches,
            topo=topo,
            targets=targets,
            rng=rng,
            data_specs=data_specs,
            return_tuple=False)
        self.X = None
        # END HUGE HACK
        
        return CopyingConcatenatingIterator(base_iterator, how_many = batch_size)

        #ret = foo
        #if return_tuple:
        #    return (ret,)
        #else:
        #    return ret
        



class CopyingConcatenatingIterator(object):

    '''Concatentates examples from the underlying iterator by copying memory (so not so efficient). Perhaps a poorly chosen name, because the concatenation is done by adding a dimension. TODO: maybe rename.'''
    def __init__(self, underlying_iterator, num_concat, return_tuple):
        self._underlying_iterator = underlying_iterator
        assert num_concat > 0, 'num_concat must be positive'
        self._num_concat = num_concat
        self._return_tuple = return_tuple
        
    def __iter__(self):
        return self

    def next(self):

        for ii in xrange(self._num_concat):
            block = self._underlying_iterator.next()
            if ii == 0:
                ret = np.zeros((self._num_concat,) + block.shape,
                               dtype = block.dtype)
            ret[ii] = block
            
        #print tmp
        #
        #pdb.set_trace()

        #to_cat = []
        #for ii in xrange(self._num_concat):
        #    to_cat.append(self._underlying_iterator.next())

        if self._return_tuple:
            return (ret,)
        else:
            return ret

    @property
    def batch_size(self):
        # if concatenation were flat
        #ret = self._underlying_iterator.batch_size * self._num_concat
        # since concatenation adds a dimension
        ret = self._num_concat
        return ret
    
    @property
    def num_batches(self):
        # rounds down
        return self._underlying_iterator.num_batches / self._num_concat
    
    @property
    def num_examples(self):
        # only changes due to rounding
        return self.batch_size * self.num_batches
    
    @property
    def uneven(self):
        print 'Not sure what this should do...'
        pdb.set_trace()
        return False    # ???
    
    @property
    def stochastic(self):
        return self._underlying_iterator.stochastic

