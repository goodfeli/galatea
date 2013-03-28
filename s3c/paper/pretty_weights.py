import sys

ignore, model_path = sys.argv

from pylearn2.utils import serial

model = serial.load(model_path)

from pylearn2.config import yaml_parse
from pylearn2.datasets import control

control.push_load_data(False)

dataset = yaml_parse.load(model.dataset_yaml_src)


W = model.W.get_value()

T = dataset.get_topological_view(W.T)

from pylearn2.gui.patch_viewer import PatchViewer

pv1 = PatchViewer((3,3),(32,32),is_color = True)

pv2 = PatchViewer((3,4),(32,32),is_color=True)

import numpy as np
rng = np.random.RandomState([1,2,3])

for i in xrange(12):
    print i
    while True:
        print 'looping'
        idxs = rng.randint(0,T.shape[0],(9,))
        for j in xrange(9):
            pv1.add_patch(T[idxs[j],:],activation=0.)
        pv1.show()
        x = raw_input('use which? (0-9): ')
        idx = idxs[eval(x)]
        break
    pv2.add_patch(T[idx,:],activation=0.)

pv2.show()
