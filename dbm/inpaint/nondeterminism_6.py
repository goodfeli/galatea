from pylearn2.devtools.record import RecordMode
from pylearn2.devtools.record import Record
from collections import OrderedDict
from pylearn2.devtools import disturb_mem
import numpy as np
import theano
from pylearn2.utils import sharedX
from theano.printing import var_descriptor

def run():
    disturb_mem.disturb_mem()


    b = sharedX(np.zeros((2,)))
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
        if var.name is not None:
            if var.name[0] != 's' or len(var.name) != 2:
                var.name = None

    for key in channels:
        updates.append((s, channels[key]))
    file_path='nondeterminism_6.txt'
    mode = RecordMode(file_path=file_path,
                      replay=0)
    f = theano.function([], mode=mode, updates=updates, on_unused_input='ignore', name='f')

    """
    print 'type(f): ',type(f)
    print 'elements of f:'
    for elem in dir(f):
        print '\t',elem
    print 'type(f.fn): ',type(f.fn)
    print 'elements of f.fn:'
    for elem in dir(f.fn):
        print '\t',elem
    """

    trials = 1

    for i in xrange(trials):
        disturb_mem.disturb_mem()
        f()

    mode.record.f.flush()
    mode.record.f.close()

    mode.set_record(Record(file_path=file_path, replay=1))

    for i in xrange(trials):
        disturb_mem.disturb_mem()
        f()


# Do several trials, since failure doesn't always occur
# (Sometimes you sample the same outcome twice in a row)
for i in xrange(10):
    run()
