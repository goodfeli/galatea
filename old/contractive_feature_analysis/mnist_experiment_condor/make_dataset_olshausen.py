from pylearn2.utils import serial
from pylearn2.datasets import mnist
import numpy as N
from scipy import io

train = mnist.MNIST(which_set="train")

D = io.loadmat('X.mat')['X']

s = D.std(axis=1)

D = D[s>.224,:]

D = D[0:50000,:]

m = D.shape[0]
print D.shape
V = D.reshape(m,20,20)

P = N.zeros((m,10,10,1))

for i in xrange(m):

    cropped = V[i,:,:]

    shrunk1 = cropped[0:20:2,:] + cropped[1:20:2,:]
    shrunk2 = shrunk1[:,0:20:2] + shrunk1[:,1:20:2]

    P[i,:,:,0] = shrunk2 / 4.
#

train.set_topological_view(P)

#train.enable_compression()

serial.save('olshausen.pkl',train)
