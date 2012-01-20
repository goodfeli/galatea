#arg1: layer 1 model
#arg2: layer 2 model

import sys

l1, l2= sys.argv[1:]

from pylearn2.utils import serial
l1 = serial.load(l1)
l2 = serial.load(l2)

W1 = l1.W.get_value()
W2 = l2.weights.get_value()


from pylearn2.gui.patch_viewer import PatchViewer

from pylearn2.config import yaml_parse
dataset = yaml_parse.load(l1.dataset_yaml_src)

import numpy as np

imgs = dataset.get_weights_view(W1.T)

N1 = l1.nhid
N = l2.nhid

count = N1

pv = PatchViewer( (N, count), imgs.shape[1:3], is_color = imgs.shape[3] == 3)

for i in xrange(N):
    w = W2[:, i]

    print (w.min(), w.mean(), w.max())

    w /= np.abs(w).max()

    wa = np.abs(w)

    to_sort = zip(wa,range(N1), w )

    s = sorted(to_sort)

    for j in xrange(count):

        idx = s[N1-j-1][1]
        mag = s[N1-j-1][2]

        if mag > 0:
            act = (mag, 0)
        else:
            act = (0, -mag)

        #print (act, mag)

        pv.add_patch( imgs[idx,...], rescale = True, activation = act)

pv.show()
