"""
This module contains cost functions to use with deep Boltzmann machines
(pylearn2.models.dbm).
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import warnings

from collections import OrderedDict

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import DefaultDataSpecsMixin
from pylearn2.models import dbm
from pylearn2.models.dbm import flatten
from pylearn2.utils import safe_izip
from pylearn2.utils import safe_zip

from pylearn2.costs.dbm import VariationalPCD_VarianceReduction

