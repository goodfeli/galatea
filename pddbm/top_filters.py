#arg1: layer 1 model
#arg2: layer 2 model

import sys
from pylearn2.utils import serial
import numpy as np

if len(sys.argv) == 3:
    l1, l2= sys.argv[1:]
    l1 = serial.load(l1)
    dataset_yaml_src = l1.dataset_yaml_src
    l2 = serial.load(l2)
else:
    model = serial.load(sys.argv[1])
    dataset_yaml_src = model.dataset_yaml_src
    l1 = model.s3c
    l2 = model.dbm.rbms[0]


W1 = l1.W.get_value()
l2_weights ,= l2.transformer.get_params()
W2 = l2_weights.get_value()

print 'Sorting so largest-norm layer 2 weights are plotted at the top'
norms = np.square(W2).sum(axis=0)
idxs = [elem[1] for elem in sorted( zip( norms, range(norms.shape[0]) ) ) ]

new = W2.copy()

for i in xrange(len(idxs)):
    new[:,i] = W2[:,idxs[i]]
W2 = new




from pylearn2.gui.patch_viewer import PatchViewer

from pylearn2.config import yaml_parse
dataset = yaml_parse.load(dataset_yaml_src)

import numpy as np

imgs = dataset.get_weights_view(W1.T)

N1 = l1.nhid
N = l2.nhid

thresh = .9
max_count = 0
total_counts = 0.
for i in xrange(N):
    w = W2[:,i]

    wa = np.abs(w)

    total = wa.sum()

    s = np.asarray(sorted(wa))

    count = 1

    while s[-count:].sum() < thresh * total:
        count += 1

    if count > max_count:
        max_count = count

    total_counts += count
ave = total_counts / float(N)

print 'average needed filters',ave

count = max_count

print 'It takes',count,'of',N1,'elements to account for ',(thresh*100.),'\% of the weight in at least one filter'

lim = 10
if count > lim:
    count = lim
    print 'Only displaying ',count,' elements though.'

if count > N1:
    count = N1

pv = PatchViewer( (N, count), imgs.shape[1:3], is_color = imgs.shape[3] == 3)

for i in xrange(N):
    w = W2[:, i]

    wneg = w[w < 0.]
    wpos = w[w > 0.]

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

        pv.add_patch( imgs[idx,...], rescale = True, activation = act)

pv.show()
