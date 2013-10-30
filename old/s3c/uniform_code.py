#Trying to come up with a binary-gaussian RBM that
#represents U([0,1]) and has very high I(v;h)
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
from theano import function
from theano import config
config.floatX = 'float64'

delta_x = 1e-2
nh = 1
sd = (1./nh)/2.
print 'sd ',sd
beta = 1./(sd ** 2.)
print 'beta ',beta
w = np.ones(nh)*2.*beta/float(nh)
print 'w ',w
b = - np.cast[str(w.dtype)](np.arange(1,nh+1)) * w #+ 50. * w
print 'b ',b


x= np.arange(0.,1.,delta_x)

zv = T.matrix()
sv = T.nnet.softplus(zv)
print 'compiling softplus'
softplus = function([zv],sv)
print 'done compiling'

def free_energy(x):
    softplusy_term = softplus(np.outer(x,w)+b).sum(axis=1)
    #return beta * (x**2.) - softplusy_term
    return softplusy_term

def unnormalized_prob(x):
    return free_energy(x)
    #arg =  -free_energy(x)
    #arg -= arg.mean()
    #return np.exp(arg)

p_tilde = unnormalized_prob(x)

#assert not np.any(np.isinf(p_tilde))

plt.plot(x,p_tilde)


plt.show()
