__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2011, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

#Included so that old imports using this filename
#still work. All development on these classes should
#be done in pylearn2
from pylearn2.models.s3c import S3C
from pylearn2.models.s3c import E_Step
from pylearn2.models.s3c import Grad_M_Step

#included to make old pkl files load
Split_E_Step = E_Step

""" This file now exists just so that imports using this
filename work. This is so that pre-existing pkl files will
continue to work. Please do all development in pylearn2.models.s3c
from now on. """
