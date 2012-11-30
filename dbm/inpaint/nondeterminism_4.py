from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.devtools import disturb_mem
import numpy as np
import theano
from pylearn2.utils import sharedX

def run(replay):
    disturb_mem.disturb_mem()

    mode = RecordMode(file_path= "nondeterminism_4.txt",
                      replay=replay)

    b = sharedX(np.zeros((2,)))
    channels = OrderedDict()

    disturb_mem.disturb_mem()

    v_max = b.max(axis=0)
    v_min = b.min(axis=0)
    v_mean = b.mean(axis=0)
    v_range = v_max - v_min

    for key, val in [
            ('max_x.max_u', v_max.max()),
            ('max_x.min_u', v_max.min()),
            ('min_x.max_u', v_min.max()),
            ('range_x.max_u', v_range.max()),
            ('mean_x.max_u', v_mean.max()),
            ]:
        disturb_mem.disturb_mem()
        channels[key] = val

    updates = []
    for key in channels:
        s = sharedX(0.)
        updates.append((s, channels[key]))
    X = theano.tensor.matrix()
    f = theano.function([X], mode=mode, updates=updates, on_unused_input='ignore', name='f')
    disturb_mem.disturb_mem()
    f(np.zeros((2,2)).astype(X.dtype))

    mode.record.f.flush()
    mode.record.f.close()

run(0)
run(1)
