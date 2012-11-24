__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import numpy as np
import warnings
from pylearn2.utils import sharedX
from pylearn2.utils import safe_zip
from theano import config
from theano import function
import theano.tensor as T
import sys
from pylearn2.devtools.record import Record

_, replay = sys.argv
if replay in ['0', '1']:
    replay = int(replay)
else:
    assert False

record_mode = Record('nondeterminism_record.txt', replay=replay)

def allocate_random():
    # Allocate a time-dependent amount of objects to increase
    # chances of all subsequent objects' ids changing from run
    # to run
    global l
    from datetime import datetime
    now = datetime.now()
    ms = now.microsecond
    ms = int(ms)
    n = ms % 1000
    m = ms / 1000
    l = [[0]*m for i in xrange(n)]
allocate_random()

params = [sharedX(np.zeros((10,)), name='param_0'),
           sharedX(np.zeros((10,10)), name='param_1'),
           sharedX(np.zeros((10,10)), name='param_2'),
           sharedX(np.zeros((10,10)), name='param_3'),
           sharedX(np.zeros((10,10)), name='param_4'),
           sharedX(np.zeros((10,10)), name='param_5'),
           sharedX(np.zeros((10,10)), name='param_6'),
           sharedX(np.zeros((10,10)), name='param_7'),
           sharedX(np.zeros((10,10)), name='param_8'),
           sharedX(np.zeros((10,10)), name='param_9'),
           sharedX(np.zeros((10,10)), name='param_10'),
           sharedX(np.zeros((10,10)), name='param_11'),
           sharedX(np.zeros((10,10)), name='param_12'),
           ]
allocate_random()

param_to_grad_shared = {}
allocate_random()

for param in params:
    allocate_random()
    param_name = param.name
    grad_name = 'BatchGradientDescent.grad_' + param_name
    grad_shared = sharedX( param.get_value() * 0., name=grad_name)
    param_to_grad_shared[param] = grad_shared


allocate_random()
norm = T.scalar()


allocate_random()
grad_shared = param_to_grad_shared.values()

grad_to_old_grad = {}
allocate_random()
for elem in grad_shared:
    allocate_random()
    grad_to_old_grad[elem] = sharedX(elem.get_value(), 'old_'+elem.name)
allocate_random()
_store_old_grad = function([], updates = dict([(grad_to_old_grad[grad], grad)
                for grad in grad_to_old_grad]), mode=record_mode, name='BatchGradientDescent._store_old_grad')

allocate_random()
_store_old_grad()
