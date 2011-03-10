# Third-party imports
import numpy

# Local imports
from framework.base import Block

class Concat(Block):
    """
    Concatenate several Block layers' output into one large final vector
    """

    def __init__(self, conf, layers = None):
        """
        Parameters in conf:

        :type k: int
        :param k: number of first representations to exclude from the concatenation
        
        0 <= k <= D means "keep all the representations but the k first"
        -D <= k < 0 means "keep the last -k representations"
        """
        # Default value for block_list
        if layers == None:
            self._layers = []
        else:
            self._layers = layers
        
        # Number of layers to keep
        kMax = len(self._layers)
        self.k = conf.get("concat_k", kMax);
        assert self.k >= -kMax and self.k <= kMax
        
        # If k < 0, then count from top instead of bottom
        if self.k < 0:
            self.k += kMax + 1
        
    def __call__(self, inputs):
        """
        Forward propagate (symbolic) input through this module, obtaining
        a representation to pass on to layers above.
        """
        transformed = inputs
        # Pass the input through each layer of the hierarchy.
        for layer in self._layers[:self.k]:
            transformed = layer(transformed)
            
        # From now on, concatenate the several representations
        representations = [transformed]
        for layer in self._layers[self.k:]:
            transformed = layer(transformed)
            representations.append(transformed)
            
        # Return the concatenations of the last representations
        return numpy.hstack(representations)
