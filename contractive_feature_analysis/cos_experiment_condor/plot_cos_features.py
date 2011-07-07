grid_rows = 3
grid_cols = 1

patch_rows = 800
patch_cols = 3200

thresh = .1

import numpy as N
patch_width = 4.5 * N.pi


import sys
from pylearn2.utils import serial
import SkyNet
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.gui import graph_2D
from pylearn2.datasets.cos_dataset import CosDataset
import theano.tensor as T
from theano import config, function, shared
floatX = config.floatX

job_name = sys.argv[1]
SkyNet.set_job_name(job_name)

components = SkyNet.get_dir_path('components')
W = shared(serial.load( components + '/W.pkl'))
whitener = serial.load( components + '/whitener.pkl')
fe = serial.load( components + '/fe.pkl')

d = CosDataset()

g = graph_2D.Graph2D(shape = (patch_rows, patch_cols),
	             xlim = (-patch_width / 2., patch_width / 2.),
                 ycenter = 0.0)

def normalize_pdf(X):
    if X.min() < 0.0:
        raise ValueError('pdf had a value of '+str(X.min())+' in it')
    X /= X.max()
    return X

def normalize_feature(X):
    X /= N.abs(X).max()

    X += 1.0
    X /= 2.0

    return X

#g.components.append( graph_2D.HeatMap( f = d.pdf, normalizer = normalize_pdf) )

pv = PatchViewer( grid_shape = (grid_rows, grid_cols), patch_shape = (patch_rows, patch_cols), is_color = True )


idx = shared(0)
X = T.matrix()
feat = fe(X)
feat.name = 'feat'
w = whitener(feat)
w.name = 'w'
final = T.dot(w,W[:,idx])

raw_f = function([X],final)

def f(X):
    mask = d.pdf(X) > thresh
    subX = X[mask,:]
    rval = N.zeros(X.shape[0],dtype=floatX)
    rval[mask] = raw_f(subX)
    return rval


for i in xrange(grid_rows * grid_cols):
    print 'filter %d' %i
    idx.set_value(i)

    g.components.append( graph_2D.HeatMap( f = f, normalizer = normalize_feature, render_mode = 'o' ) )

    patch = g.render()

    pv.add_patch(patch * 2.0 -1.0, rescale = False)

    del g.components[-1]

pv.show()
