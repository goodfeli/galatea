from pylearn2.utils.serial import load
import sys
import theano.tensor as T
import numpy as np

print 'loading model'
model_path = sys.argv[1]
model = load(model_path)
model.make_pseudoparams()

print 'loading dataset'
dataset_path = sys.argv[2]
if dataset_path.endswith('.pkl'):
    dataset = load(dataset_path)
else:
    from pylearn2.config import yaml_parse
    dataset = yaml_parse.load_path(dataset_path)
XS = dataset.get_design_matrix()


X = T.matrix()

H = model.e_step.variational_inference(X)['H_hat']

ipt = T.dot(X,model.W)

from theano import function

f = function([X],[ ipt, H])

import matplotlib.pyplot as plt

for i in xrange(model.nhid):
    print 'feature ',i

    ipts = []
    Hs = []

    batch_size = 5000
    for j in xrange(0,100000,batch_size):
        print '\t',j
        iptj, Hj = f(XS[j:j+batch_size,:])
        ipts.append(iptj[:,i])
        Hs.append(Hj[:,i])

    ipt = np.concatenate(ipts,axis=0)
    H = np.concatenate(Hs,axis=0)

    plt.scatter(ipt,H)

    plt.show()

    print 'waiting'
    x = raw_input()
    if x == 'q':
        quit()
    print 'running'
