from pylearn2.utils.serial import load
import sys

print 'loading model'
model_path = sys.argv[1]
model = load(model_path)
model.make_pseudoparams()

print 'loading dataset'
dataset_path = sys.argv[2]
dataset = load(dataset_path)
X = dataset.get_design_matrix()


X = X[0:5000,:]

H = model.e_step.mean_field(X)['H']

import theano.tensor as T
ipt = T.dot(X,model.W)

from theano import function

ipt, H = function([],[ ipt, H])()

import matplotlib.pyplot as plt

for i in xrange(model.nhid):
    print 'feature ',i

    plt.scatter(ipt[:,i],H[:,i])

    plt.show()

    print 'waiting'
    x = raw_input()
    if x == 'q':
        quit()
    print 'running'
