import numpy as N
from theano import config
floatX = config.floatX
from pylearn2.datasets.cos_dataset import CosDataset
from pylearn2.gui.graph_2D import Graph2D, HeatMap
import theano.tensor as T
from theano import function
from theano.printing import Print

rows = 500
cols  = 1000

xmin = -6.2
xmax = 6.2

import sys
from pylearn2.utils import serial

model = serial.load(sys.argv[1])
model.redo_theano()
dataset = CosDataset()

print 'examples seen: '+str(model.examples_seen)

PDF = 0    #pdf
GPDF = 1   #gradient of pdf
GDPDF = 2  #gradient direction of pdf
ENERGY = 3 #energy
SCORE = 4  #score
SCORED = 5 #score direction

if len(sys.argv) == 2 or sys.argv[2] == '--pdf':
    t = PDF
elif sys.argv[2] == '--gpdf':
    t = GPDF
elif sys.argv[2] == '--gdpdf':
    t = GDPDF
elif sys.argv[2] == '--energy':
    t = ENERGY
elif sys.argv[2] == '--score':
    t = SCORE
elif sys.argv[2] == '--scored':
    t = SCORED
else:
    raise ValueError('unknown flag '+sys.argv[2])

g = Graph2D( shape = (rows, cols), xlim=(xmin,xmax), ycenter = 0. )


def energy_normalizer(I):
    I -= I.min()
    I /= I.max()
    return  (I*2.0)-1.0

def pdf_normalizer(I):
    assert I.min() >= 0.0
    I /= I.max()

    for i in xrange(I.shape[0]):
        for j in xrange(I.shape[1]):
            if N.any(N.isnan(I[i,j,:]) + N.isinf(I[i,j,:])):
                I[i,j,0] = 1.0
                I[i,j,1] = 0.0
                I[i,j,2] = 0.0

    return (I*2.0)-1.0

def gpdf_normalizer(I):
    I /= N.abs(I).max()
    return I

def gdpdf_normalizer(I):
    return I

X = T.matrix()

class FuckYouTheano:
    def __init__(self, join_should_be_able_to_do_this):
        self.f = join_should_be_able_to_do_this

    def __call__(self, X):
        rval = N.zeros((X.shape[0],3),dtype=floatX)
        rval[:,0:2] = self.f(X)
        return rval

def grad_func( pdf ):
    grad = T.grad(pdf.sum(), X)
    return FuckYouTheano(function([X],grad))

def grad_dir_func( pdf ):
    grad = T.grad(pdf.sum(), X)
    grad = Print('before',attrs=['min','max'])(grad)
    grad /= T.sqrt(1e-15+T.sum(T.sqr(grad),axis=1).dimshuffle(0,'x'))
    grad = Print('after',attrs=['min','max'])(grad)
    return FuckYouTheano(function([X],grad))


if t not in [ENERGY, SCORE, SCORED]:
    g.components.append(HeatMap( f = function([X], model.free_energy(X)),  normalizer = None ))
    offset = g.render().mean()
#

if t == ENERGY:
    df = dataset.free_energy_func
    mfe = model.free_energy(X)
    mf = function([X],mfe)
    normalizer = energy_normalizer
elif t == PDF:
    df = dataset.pdf_func

    mfe = model.free_energy(X)
    mfe = Print('model free energy',attrs=['min','max'])(mfe)
    mf = function([X], T.exp(-mfe+offset))

    normalizer = pdf_normalizer
elif t == GPDF:
    df = grad_func(dataset.pdf(X))
    mf = grad_func(T.exp(-model.free_energy(X)+offset))

    normalizer = gpdf_normalizer
elif t == GDPDF:
    df = grad_dir_func(dataset.pdf(X))
    mf = grad_dir_func(T.exp(-model.free_energy(X)+offset))

    normalizer = gdpdf_normalizer
elif t == SCORE:
    df = grad_func(- dataset.free_energy(X))
    mf = grad_func( - model.free_energy(X))

    normalizer = gpdf_normalizer
elif t == SCORED:
    df = grad_dir_func(- dataset.free_energy(X))
    mf = grad_dir_func( - model.free_energy(X))

    normalizer = gdpdf_normalizer
else:
    assert False


g.components.append(HeatMap(f = df, normalizer = normalizer))
dimg = g.render()

g.components.pop()
g.components.append(HeatMap(f = mf, normalizer = normalizer))

mimg = g.render()

from pylearn2.gui.patch_viewer import PatchViewer

pv = PatchViewer((2,1),dimg.shape[0:2], is_color = dimg.shape[2] == 3)
pv.add_patch(dimg,rescale = False)
pv.add_patch(mimg, rescale = False)
pv.show()

