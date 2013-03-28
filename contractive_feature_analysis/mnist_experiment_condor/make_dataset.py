from pylearn2.utils import serial
from pylearn2.datasets import mnist
from pylearn2.datasets import preprocessing
import numpy as N

train = mnist.MNIST(which_set="train")

D = train.get_topological_view()[:,:,:,0]

m = D.shape[0]

P = N.zeros((m,10,10,1))

for i in xrange(m):
    I = D[i,:,:]
    colsum = I.sum(axis=0)

    firstcol = 0

    while colsum[firstcol] == 0.0:
        firstcol += 1
    #

    lastcol = 27
    while colsum[lastcol] == 0.0:
        lastcol -= 1
    #

    lastcol = max(firstcol+19,lastcol)

    if lastcol > 27:
        firstcol -= lastcol - 27
        lastcol = 27
    #

    if lastcol - firstcol + 1 > 20:
        print 'example ',i
        print 'image width ',lastcol-firstcol+1
        assert False


    rowsum = I.sum(axis=1)

    firstrow = 0
    while rowsum[firstrow] == 0.0:
        firstrow += 1
    #

    lastrow = 27
    while rowsum[lastrow] == 0.0:
        lastrow -= 1
    #

    lastrow = max(firstrow + 19, lastrow)

    if lastrow > 27:
        firstrow  -= lastrow - 27
        lastrow = 27
    #

    if lastrow - firstrow + 1 > 20:
        assert False

    cropped = D[i,firstrow:lastrow+1,firstcol:lastcol+1]

    shrunk1 = cropped[0:28:2,:] + cropped[1:28:2,:]
    shrunk2 = shrunk1[:,0:28:2] + shrunk1[:,1:28:2]

    P[i,:,:,0] = shrunk2 / 4.
#

train.set_topological_view(P)



"""pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ZCA(n_drop_components = 67))


train.apply_preprocessor(preprocessor = pipeline, can_fit = True)


rng = N.random.RandomState([1,2,3])

bad = train.X.std(axis=0) == 0

def fix(dataset):
    dataset.X[:,bad] = rng.randn(dataset.X.shape[0],bad.sum()) * .3

fix(train)

train.enable_compression()

print train.X.mean(axis=0)
"""

serial.save('mnist_preprocessed_train.pkl',train)
