import sys
from pylearn2.utils import serial
import numpy as np
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.config import yaml_parse

_, model_path = sys.argv

model = serial.load(model_path)

layer_1, layer_2 = model.hidden_layers[0:2]

W1 = layer_1.get_weights()
W2 = layer_2.get_weights()
print W1.shape
print W2.shape

prod = np.dot(W1,W2)
pv = make_viewer(prod.T)
pv.show()


print 'Sorting so largest-norm layer 2 weights are plotted at the top'
norms = np.square(W2).sum(axis=0)
idxs = [elem[1] for elem in sorted( zip( -norms, range(norms.shape[0]) ) ) ]

new = W2.copy()

for i in xrange(len(idxs)):
    new[:,i] = W2[:,idxs[i]]
W2 = new


dataset_yaml_src = model.dataset_yaml_src
dataset = yaml_parse.load(dataset_yaml_src)

import numpy as np

imgs = dataset.get_weights_view(W1.T)

N1 = W1.shape[1]
N = W2.shape[1]

N = min(N,100)

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

lim = 300
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
