import sys
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
import theano.tensor as T
import numpy as np
from pylearn2.utils import sharedX

batch_size = 100
batches = 50

ignore, model_path = sys.argv

print 'loading model...'
model = serial.load(model_path)
model.make_pseudoparams()

print 'loading dataset...'
dataset  = yaml_parse.load(model.dataset_yaml_src)

print 'compiling update mean'
V = T.matrix()
obs = model.inference_procedure.infer(V)

cur_H_mean = T.mean(obs['H_hat'],axis=0)

n = model.s3c.nhid
H_mean = sharedX(np.zeros(n,))

update_mean = function([V], updates = { H_mean : H_mean + cur_H_mean })

print 'compiling update cov'
Z = obs['H_hat'] - H_mean
cov = sharedX(np.zeros((n,n)))

update_cov = function([V], updates = { cov : cov + T.dot(Z.T,Z) / np.cast['float32'](batch_size) } )


for i in xrange(batches):
    print 'learning mean batch',i
    X = dataset.get_batch_design(batch_size)
    update_mean(X)

H_mean.set_value(H_mean.get_value() / float(batches))

for i in xrange(batches):
    print 'updating cov',i
    X = dataset.get_batch_design(batch_size)
    update_cov(X)

cov.set_value(cov.get_value() / float(batches))


print 'saving to cov.npy...'
serial.save('cov.npy',cov.get_value())
