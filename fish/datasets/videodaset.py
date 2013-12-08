"""
Modified from:
     lisa_emotiw / emotiw / common / datasets / faces / facetubes.py
"""
# Basic Python packages
import functools

# External dependencies
import numpy as np

# In-house dependencies
import theano
from theano import config
from theano.gof.op import get_debug_values
from theano.sandbox.cuda.type import CudaNdarrayType
from theano.tensor import TensorType

from pylearn2.datasets import Dataset
from pylearn2.utils import safe_zip
from pylearn2.utils.data_specs import is_flat_specs
from pylearn2.utils.iteration import (FiniteDatasetIterator,
                                      resolve_iterator_class)
from pylearn2.space import CompositeSpace, Space, VectorSpace, Conv2DSpace
#from emotiw.scripts.mirzamom.conv3d.space import Conv3DSpace
# Current project


class VideoDataset(Dataset):
    def get_data(self):
        return self.data

    def get_data_specs(self):
        return self.data_specs

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            raise ValueError('iteration mode not provided and no default '
                             'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        if data_specs is None:
            data_specs = getattr(self, '_iter_data_specs', None)

        # TODO: figure out where to to the scaling more cleanly.
        def list_to_scaled_array(batch):
            # batch is either a 4D ndarray, or a list of length 1
            # containing a 4D ndarray. Make it a 5D ndarray,
            # with shape 1 on the first dimension.
            # Also scale it from [0, 255] to [0, 1]
            if isinstance(batch, list):
                assert len(batch) == 1
                batch = batch[0]
            batch = batch.astype(config.floatX)
            batch /= 255.
            return batch[np.newaxis]

        convert_fns = []
        for space in data_specs[0].components:
            if (isinstance(space, FaceTubeSpace) and
                    space.axes[0] == 'b'):
                convert_fns.append(list_to_scaled_array)
            else:
                convert_fns.append(None)

        return FiniteDatasetIteratorVariableSize(
                self,
                mode(self.n_samples,
                     batch_size,
                     num_batches,
                     rng),
                data_specs=data_specs,
                return_tuple=return_tuple,
                convert_fns=convert_fns)


class FiniteDatasetIteratorVariableSize(FiniteDatasetIterator):
    def __init__(self, dataset, subset_iterator, data_specs=None,
                 return_tuple=False, convert_fns=None):
        """
        convert_fns: function or tuple of function, organized as
            in data_specs, to be applied on the raw batches of
            data. "None" can be used as placeholder for the identity.
        """
        self._deprecated_interface = False
        if data_specs is None:
            raise TypeError("data_specs not provided")
        self._data_specs = data_specs
        self._dataset = dataset
        self._subset_iterator = subset_iterator
        self._return_tuple = return_tuple

        # Keep only the needed sources in self._raw_data.
        # Remember what source they correspond to in self._source
        assert is_flat_specs(data_specs)

        dataset_space, dataset_source = self._dataset.get_data_specs()
        assert is_flat_specs((dataset_space, dataset_source))

        # the dataset's data spec is either a single (space, source) pair,
        # or a pair of (non-nested CompositeSpace, non-nested tuple).
        # We could build a mapping and call flatten(..., return_tuple=True)
        # but simply putting spaces, sources and data in tuples is simpler.
        if not isinstance(dataset_source, tuple):
            dataset_source = (dataset_source,)

        if not isinstance(dataset_space, CompositeSpace):
            dataset_sub_spaces = (dataset_space,)
        else:
            dataset_sub_spaces = dataset_space.components
        assert len(dataset_source) == len(dataset_sub_spaces)

        all_data = self._dataset.get_data()
        if not isinstance(all_data, tuple):
            all_data = (all_data,)

        space, source = data_specs
        if not isinstance(source, tuple):
            source = (source,)
        if not isinstance(space, CompositeSpace):
            sub_spaces = (space,)
        else:
            sub_spaces = space.components
        assert len(source) == len(sub_spaces)

        self._raw_data = tuple(all_data[dataset_source.index(s)]
                               for s in source)
        self._source = source

        if convert_fns is None:
            self._convert = [None for s in source]
        else:
            if not isinstance(convert_fns, (list, tuple)):
                convert_fns = (convert_fns,)
            assert len(convert_fns) == len(source)
            self._convert = list(convert_fns)

        for i, (so, sp) in enumerate(safe_zip(source, sub_spaces)):
            idx = dataset_source.index(so)
            dspace = dataset_sub_spaces[idx]

            # Compose the functions
            fn = self._convert[i]
            needs_cast = not (self._raw_data[i][0].dtype == config.floatX)
            if needs_cast:
                if fn is None:
                    fn = lambda batch: np.cast[config.floatX](batch)
                else:
                    fn = (lambda batch, prev_fn=fn:
                          np.cast[config.floatX](prev_fn(batch)))

            needs_format = not sp == dspace
            if needs_format:
                # "dspace" and "sp" have to be passed as parameters
                # to lambda, in order to capture their current value,
                # otherwise they would change in the next iteration
                # of the loop.
                if fn is None:
                    fn = (lambda batch, dspace=dspace, sp=sp:
                          dspace.np_format_as(batch, sp))
                else:
                    fn = (lambda batch, dspace=dspace, sp=sp, fn_=fn:
                          dspace.np_format_as(fn_(batch), sp))

            self._convert[i] = fn
