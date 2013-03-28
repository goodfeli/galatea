import sys
from pylearn2.utils import serial
from pylearn2.datasets.mnist import MNIST
import theano.tensor as T
import warnings
from theano import function
import numpy as np
from scipy import io

ignore, model_path = sys.argv

print 'loading model'
model = serial.load(model_path)
model.make_pseudoparams()

#if model.dbm.num_classes > 0:
#    warnings.warn("mutilating the model in the same way Russ did")
#    model.dbm.num_classes = 0

has_class = model.dbm.num_classes > 0

dataset = MNIST(which_set = 'train', one_hot = True)

print 'compiling inference function'
X = T.matrix()
X.tag.test_value = np.zeros((2,784),dtype='float32')

supervised = model.dbm.num_classes > 0

if has_class:
    Y = -1
else:
    Y = None

obs = model.inference_procedure.infer(X,Y)

H = obs['H_hat']
G = obs['G_hat']
assert len(G) == 1
G ,= G
if has_class:
    Y = obs['Y_hat']

H_list = []
G_list = []

outputs = [H,G]
if has_class:
    outputs.append(Y)
    Y_list = []

batch_size = 100

f = function([X],outputs)

for i in xrange(0,dataset.X.shape[0],batch_size):
    print 'processing example',i
    Xv = dataset.X[i:i+batch_size,:]
    if has_class:
        Hv, Gv, Yv = f(Xv)
        Y_list.append(Yv)
    else:
        Hv, Gv = f(Xv)

    H_list.append(Hv)
    G_list.append(Gv)

print 'concatenating'
H = np.concatenate(H_list, axis = 0)
assert len(H.shape) == 2
assert H.shape[0] == dataset.X.shape[0]

G = np.concatenate(G_list, axis = 0)
assert len(G.shape) == 2
assert G.shape[0] == dataset.X.shape[0]

y = dataset.y
d = { 'X': dataset.X, 'H' : H, 'G' : G, 'y' : y }
if has_class:
    Y = np.concatenate(Y_list, axis =0 )
    assert Y.shape[0] == dataset.X.shape[0]
    assert Y.shape[1] == 10
    d['Y'] = Y


batchdata = np.zeros((100,784,600))
batchtargets = np.zeros((100,10,600))
temp_h2_train = np.zeros((100,1000,600))
for i in xrange(600):
    batchdata[:,:,i] = dataset.X[i*100:(i+1)*100,:]
    batchtargets[:,:,i] = dataset.y[i*100:(i+1)*100,:]
    temp_h2_train[:,:,i] = G[i*100:(i+1)*100,:]


dataset = MNIST( which_set = 'test', one_hot = True)

H_list = []
G_list = []
X_list = []
Y_list = []

for i in xrange(0,dataset.X.shape[0],batch_size):
    print 'processing example',i
    Xv = dataset.X[i:i+batch_size,:]
    if has_class:
        Hv, Gv, Yv = f(Xv)
        Y_list.append(Yv)
    else:
        Hv, Gv = f(Xv)

    H_list.append(Hv)
    G_list.append(Gv)
    X_list.append(Xv)

print 'concatenating'
H = np.concatenate(H_list, axis = 0)
assert len(H.shape) == 2
assert H.shape[0] == dataset.X.shape[0]

G = np.concatenate(G_list, axis = 0)
assert len(G.shape) == 2
assert G.shape[0] == dataset.X.shape[0]

X = np.concatenate(X_list, axis = 0)
if has_class:
    Y = np.concatenate(Y_list,axis = 0)

print 'saving'

assert X.shape[0] == 10000
assert X.shape[1] == 784
y = dataset.y

testbatchdata = np.zeros((100,784,100))
testbatchtargets = np.zeros((100,10,100))
temp_h2_test = np.zeros((100,1000,100))
for i in xrange(100):
    testbatchdata[:,:,i] = dataset.X[i*100:(i+1)*100,:]
    testbatchtargets[:,:,i] = dataset.y[i*100:(i+1)*100,:]
    temp_h2_test[:,:,i] = G[i*100:(i+1)*100,:]

vishid = model.s3c.W.get_value() * model.s3c.mu.get_value()
hidpen = model.dbm.W[0].get_value()

d = { 'batchdata': batchdata,
      'testbatchdata' : testbatchdata,
      'batchtargets' : batchtargets,
      'testbatchtargets' : testbatchtargets,
      'temp_h2_train' : temp_h2_train,
      'temp_h2_test' : temp_h2_test,
      'vishid' : vishid,
      'hidpen' : hidpen,
      'hidbiases' : model.s3c.bias_hid.get_value(),
      'penbiases' : model.dbm.bias_hid[0].get_value()
   }
io.savemat('pylearn2_dump.mat',d)


