"""Investigating the effect of projecting a gradient into
the tangent space of a hypersphere. Regardless of the true
problem size we can treat this as two dimensional: axis 0
is the direction of the current parameter vector, and axis
1 is the direction of the component of the gradient update
that is orthogonal to this direction. """

import numpy as np
import matplotlib.pyplot as plt

#current value of the parameter. always the same due to
#the definition of our two axes and the norm constraint
x = np.asarray([1,0])

def norm(v):
    return np.sqrt(np.square(v).sum())

def final_steps(g1,g2,alpha):
    """
    g1: magnitude of component of gradient in same direction as current param vector
    g2: magnitude of component of gradient in direction orthogonal to current param vector
    alpha: learning rate
    returns:
       n1: new param vector after doing gradient step and reprojecting to unit sphere
       n2: new param vector after projecting gradient into tangent of sphere, doing gradient step, and reprojecting to unit sphere
    """

    g = np.asarray([g1,g2])
    h = np.asarray([0,g2])

    s1 = x + alpha * g
    s2 = x + alpha * h


    n1 = s1 / norm(s1)
    n2 = s2 / norm(s2)
    return n1, n2

def dratio(g1,g2,alpha):
    """ returns ratio of length of step with projection to length of step without projection """
    n1, n2 = final_steps(g1, g2, alpha)

    d1 = norm(x - n1)
    d2 = norm(x - n2)

    return d2/d1

def dotprod(g1,g2,alpha):
    """ returns the dot product between the two resulting steps """
    n1, n2 = final_steps(g1, g2, alpha)

    return np.dot(n1,n2)


alpha = .01
g2 = 1.

g1 = np.arange(-10.,10.,.01)

y = np.asarray([ dratio(g1[i],g2,alpha) for i in xrange(g1.shape[0]) ] )

plt.plot(g1,y)

plt.show()
