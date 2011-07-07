import numpy as N
from theano import config
floatX = config.floatX

rows = 500
cols  = 1000

xmin = -6.2
xmax = 6.2

ymin = -3.3
ymax = 3.3

from pylearn2.config import yaml_parse
import sys
from pylearn2.utils import serial

model = serial.load(sys.argv[1])
model.redo_theano()
dataset = yaml_parse.load(model.dataset_yaml_src)


PDF = 0    #pdf
GPDF = 1   #gradient of pdf
GDPDF = 2  #gradient direction of pdf
ENERGY = 3 #energy

if len(sys.argv) == 2 or sys.argv[2] == '--pdf':
    t = PDF
elif sys.argv[2] == '--gpdf':
    t = GPDF
elif sys.argv[2] == '--gdpdf':
    t = GDPDF
elif sys.argv[2] == '--energy':
    t = ENERGY



def make_img(f):
    rval = N.zeros((rows,cols,1))

    for i in xrange(cols):
        print 'col '+str(i)
        x = xmin + (xmax-xmin)*float(i)/float(cols-1)
        ys = [ymin+(ymax-ymin)*float(j)/float(rows-1) for j in xrange(rows) ]
        assert len(ys) == rows

        xs = N.zeros((rows,1))+x
        assert xs.shape[0] == rows
        assert xs.shape[1] == 1

        ys = N.asarray(ys)
        ys = N.asarray([ys]).T
        assert ys.shape[0] == rows
        assert ys.shape[1] == 1

        mat = N.hstack( (xs,ys) )

        mat = N.cast[floatX](mat)

        assert mat.shape[0] == rows
        assert mat.shape[1] == 2

        mf = f(mat)

        assert mf.shape[0] == 500
        assert len(mf.shape) == 1

        #try:
        rval[:,i,0] = mf
        #except ValueError, e:
        #    print rval[:,i,0].shape
        #    print f(mat).shape
        #    raise e

    assert not N.any(N.isinf(rval))
    assert not N.any(N.isnan(rval))

    #rval -= rval.mean()
    #return rval

    if t == ENERGY:
        rval -= rval.min()
        rval /= rval.max()
        return rval

    assert rval.min() >= 0.0
    mx = rval.max()

    rval /= mx
    rval *= 2.0
    rval -= 1.0


    if t == PDF:
        return rval

    rval2 = N.zeros((rval.shape[0]-1,rval.shape[1]-1,3))

    for i in xrange(rval2.shape[0]):
        for j in xrange(rval2.shape[1]):
            #rval2[i,j,0] = N.sign(rval[i+1,j,0]-rval[i,j,0])
            #rval2[i,j,1] = N.sign(rval[i,j+1,0]-rval[i,j,0])

            rval2[i,j,0] = rval[i+1,j,0]-rval[i,j,0]
            rval2[i,j,1] = rval[i,j+1,0]-rval[i,j,0]
            if t == GDPDF:
                rval2[i,j,:] /= (1e-15+N.sqrt((rval2[i,j,:] ** 2.).sum()))
            #
        #
    #

    if t == GPDF:
        rval2 /= N.abs(rval2).max()


    return rval2

def pdf(mat):
    rval = model.E_X_batch_func(mat)
    #assert not N.any(N.isinf(rval))
    #assert not N.any(N.isnan(rval))
    assert rval.shape[0] == mat.shape[0]
    assert len(rval.shape) == 1

    #print (rval.min(),rval.max())

    if t != ENERGY:
        rval = N.exp(-rval)
    #assert not N.any(N.isinf(rval))
    #assert not N.any(N.isnan(rval))

    return rval



print 'making dataset image'
if t == ENERGY:
    dimg = make_img(dataset.energy)
else:
    dimg = make_img(dataset.pdf)
print 'making model image'
mimg = make_img(pdf)

mn = min([dimg.min(),mimg.min()])
dimg -= mn
mimg -= mn
mx = max([dimg.max(),mimg.max()])
dimg /= mx
mimg /= mx

dimg *= 2.
dimg -= 1.
mimg *= 2.
mimg -= 1.


from pylearn2.gui.patch_viewer import PatchViewer

pv = PatchViewer((2,1),dimg.shape[0:2], is_color = dimg.shape[2] == 3)
pv.add_patch(dimg,rescale = False)
pv.add_patch(mimg, rescale = False)
pv.show()

