import numpy as N
import matplotlib.pyplot as plt
import serialutil

X = serialutil.load('tsne_for_yann.pkl')

assert X.shape[1] == 2

ytrain = N.zeros((4000,3))
yvalid = N.zeros((4096,3))
ytest = N.zeros((4096,3))
ytrain[:,0] = 1.
yvalid[:,1] = 1.
ytest[:,2] = 1.

y = N.concatenate((ytrain,yvalid,ytest))

plt.scatter(x = X[:,0], y= X[:,1], c = y)

plt.show()

