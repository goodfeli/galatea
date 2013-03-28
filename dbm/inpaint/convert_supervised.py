from scipy import io
from pylearn2.utils import serial
from galatea.dbm.inpaint.super_dbm import MLP_Wrapper
d = io.loadmat('../code_DBM/backprop_weights.mat')

for key in d:
    try:
        d[key] = d[key].astype('float32')
    except:
        pass

w1_vishid = d['w1_vishid']
w1_penhid = d['w1_penhid']
w2 = d['w2']
w_class = d['w_class']
h1_biases = d['h1_biases']
h2_biases = d['h2_biases']
topbiases = d['topbiases']
test_err = d['test_err']

assert w1_vishid.shape == (784, 500)
assert w1_penhid.shape == (1000, 500)
assert w2.shape == (500, 1000)
assert w_class.shape == (1000, 10)
assert h1_biases.shape == (1,500)
assert h2_biases.shape == (1,1000)
assert topbiases.shape == (1,10)


print 'test_err was',float(test_err[-1][-1])/10000.

dbm = serial.load('russ/fullmnist_dbm.pkl')

model = MLP_Wrapper(dbm)

model.vishid.set_value(w1_vishid)
model.penhid.set_value(w1_penhid)
model.hidpen.set_value(w2)
model.hidbias.set_value(h1_biases[0,:])
model.penbias.set_value(h2_biases[0,:])
model.c.set_weights(w_class)
model.c.set_biases(topbiases[0,:])
model.dataset_yaml_src = dbm.dataset_yaml_src

serial.save('russ/after_sup_train.pkl', model)
