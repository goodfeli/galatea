import sys
from pylearn2.utils import serial
import numpy as np

ignore, model_path = sys.argv


model = serial.load(model_path)

if hasattr(model,'s3c'):
    s3c = model.s3c
else:
    s3c = model

W = s3c.W

Wv = W.get_value()

to_sort = []

for i in xrange(Wv.shape[1]-1):
    for j in xrange(i+1,Wv.shape[1]):
        dot = abs(np.dot(Wv[:,i],Wv[:,j]))
        to_sort.append( (-dot, (i,j) ) )


to_sort = sorted(to_sort)[0:100]

print -to_sort[0][0]
print -to_sort[99][0]

from pylearn2.config import yaml_parse

dataset = yaml_parse.load(model.dataset_yaml_src)

weights_view = dataset.get_weights_view(Wv.T)

from pylearn2.gui.patch_viewer import PatchViewer

pv = PatchViewer((100,2),(28,28),is_color=False)

for i in xrange(100):
    l, r = to_sort[i][1]
    l = weights_view[l]
    r = weights_view[r]
    pv.add_patch(l,rescale=True)
    pv.add_patch(r,rescale=True)

pv.show()
