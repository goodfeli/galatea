from pylearn2.devtools.record import RecordMode
from collections import OrderedDict
from pylearn2.devtools import disturb_mem
import numpy as np
import theano
from pylearn2.utils import sharedX
from theano.printing import var_descriptor

def run(replay):
    disturb_mem.disturb_mem()

    mode = RecordMode(file_path= "nondeterminism_4.txt",
                      replay=replay)

    b = sharedX(np.zeros((2,)), name='b')
    channels = OrderedDict()

    disturb_mem.disturb_mem()

    v_max = b.max(axis=0)
    v_min = b.min(axis=0)
    v_range = v_max - v_min

    updates = []
    for i, val in enumerate([
            v_max.max(),
            v_max.min(),
            v_range.max(),
            ]):
        disturb_mem.disturb_mem()
        s = sharedX(0., name='s_'+str(i))
        updates.append((s, val))

    for var in theano.gof.graph.ancestors(update for var, update in updates):
        if var.name is not None and var.name is not 'b':
            if var.name[0] != 's' or len(var.name) != 2:
                var.name = None

    for key in channels:
        updates.append((s, channels[key]))
    f = theano.function([], mode=mode, updates=updates, on_unused_input='ignore', name='f')
    for output in f.maker.fgraph.outputs:
        mode.record.handle_line(var_descriptor(output)+'\n')
    disturb_mem.disturb_mem()
    f()

    mode.record.f.flush()
    mode.record.f.close()

# Do several trials, since failure doesn't always occur
# (Sometimes you sample the same outcome twice in a row)
for i in xrange(10):
    run(0)
    run(1)
