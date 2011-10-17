import sys
import numpy as np

model_path = sys.argv[1]

if len(sys.argv) > 2:
    tidx = int(sys.argv[2])
else:
    tidx = -1

from pylearn2.utils.serial import load

model = load(model_path)

monitor = model.monitor

em_functional_channels = {}

for key in monitor.channels:
    if key.startswith('em_functional_'):
        em_functional_channels[key] = monitor.channels[key]

vals = np.zeros(len(em_functional_channels.keys()))


for key in em_functional_channels:
    pieces = key.split('_')
    assert len(pieces) == 3
    idx = int(pieces[2]) - 1

    val = em_functional_channels[key].val_record[tidx]

    vals[idx] = val

if len(vals) == 0:
    print 'this model did not use monitoring of the em functional across the e step'
    quit(-1)

from matplotlib import pyplot as plt

plt.plot(vals)

plt.show()
