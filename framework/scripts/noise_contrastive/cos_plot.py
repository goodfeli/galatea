import numpy as N

rows = 500
cols  = 1000

xmin = -6.6
xmax = 6.6

ymin = -3.3
ymax = 3.3

from framework.config import yaml_parse
import sys
from framework.utils import serial

model = serial.load(sys.argv[1])
model.redo_theano()
dataset = yaml_parse.load(model.dataset_yaml_src)

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

    assert rval.min() >= 0.0
    rval /= rval.max()
    rval *= 2.0
    rval -= 1.0

    rval2 = N.zeros((rval.shape[0]-1,rval.shape[1]-1,3))

    for i in xrange(rval2.shape[0]):
        for j in xrange(rval2.shape[1]):
            #rval2[i,j,0] = N.sign(rval[i+1,j,0]-rval[i,j,0])
            #rval2[i,j,1] = N.sign(rval[i,j+1,0]-rval[i,j,0])

            rval2[i,j,0] = rval[i+1,j,0]-rval[i,j,0]
            rval2[i,j,1] = rval[i,j+1,0]-rval[i,j,0]
            rval2[i,j,:] /= (1e-15+N.sqrt((rval2[i,j,:] ** 2.).sum()))
        #
    #


    rval = rval2

    return rval

def pdf(mat):
    rval = model.E_X_batch_func(mat)
    assert rval.shape[0] == mat.shape[0]
    assert len(rval.shape) == 1
    rval = N.exp(-rval)
    return rval

print 'making dataset image'
dimg = make_img(dataset.pdf)
print 'making model image'
mimg = make_img(pdf)

from framework.gui.patch_viewer import PatchViewer

pv = PatchViewer((2,1),dimg.shape[0:2], is_color = dimg.shape[2] == 3)
pv.add_patch(dimg,rescale = False)
pv.add_patch(mimg, rescale = False)
pv.show()

