from s3c import rotate_towards
from theano import function
import theano.tensor as T
import numpy as np
from matplotlib import pyplot as plt

X = T.dmatrix()
Y = T.dmatrix()
amt = T.dscalar()

rotated = rotate_towards(X,Y,amt)

rotate = function([X,Y,amt], rotated)

D = 3

rng = np.random.RandomState([4,2,5])

while True:
    a = rng.randn(D,1)
    b = a + 0.1 * rng.randn(D,1)


    unit_a = a / np.sqrt(np.square(a).sum())
    print unit_a
    print b / np.sqrt(np.square(b).sum())
    b_component = b - np.dot(b.T,unit_a) * unit_a
    unit_b = b_component / np.sqrt(np.square(b_component).sum())

    step = .01
    amts = np.arange(0,1+step,step)

    xs = []
    ys = []

    for i in xrange(amts.shape[0]):
        amt = amts[i]

        res = rotate(unit_a, b, amt)

        if i in [0,amts.shape[0]-1]:
            print i
            print amts[i]
            print res

        xs.append(np.dot(unit_a.T, res))
        ys.append(np.dot(unit_b.T, res))


    plt.scatter(xs,ys)
    plt.show()

    print 'waiting...'
    x = raw_input()
    if x == 'q':
        break
    print 'running...'
