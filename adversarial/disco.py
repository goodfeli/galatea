from pylearn2.models.mlp import *

class Disconnector(Linear):
    """
    Rectified linear MLP layer (Glorot and Bengio 2011).

    Parameters
    ----------
    left_slope : float
        The slope the line should have left of 0.
    kwargs : dict
        Keyword arguments to pass to `Linear` class constructor.
    """

    def __init__(self, ofs = 0.1, left_slope=0.0, **kwargs):
        super(Disconnector, self).__init__(**kwargs)
        self.ofs = ofs
        self.left_slope = left_slope

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        # Original: p = p * (p > 0.) + self.left_slope * p * (p < 0.)
        # T.switch is faster.
        # For details, see benchmarks in
        # pylearn2/scripts/benchmark/time_relu.py
        p = T.switch(p > 0., p + self.ofs, self.left_slope * p)
        return p

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):

        raise NotImplementedError()
    """
    Rectified linear MLP layer (Glorot and Bengio 2011).

    Parameters
    ----------
    left_slope : float
        The slope the line should have left of 0.
    kwargs : dict
        Keyword arguments to pass to `Linear` class constructor.
    """

