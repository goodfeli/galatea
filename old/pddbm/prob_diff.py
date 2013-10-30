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
prod = sharedX(np.zeros((n,n)))

H = obs['H_hat']
update = function([V], updates = { H_mean : H_mean + cur_H_mean,
                                        prod: prod + T.dot(H.T,H) / np.cast['float32'](batch_size)})

for i in xrange(batches):
    print 'learning mean batch',i
    X = dataset.get_batch_design(batch_size)
    update(X)

"""
Let P(v,h) = P_D( v ) Q_v ( h )

P( h_j = 1 | h_i = 1) = P( h_i = 1, h_j = 1) / P( h_i = 1 )
                      = E_D[ h_hat_i h_hat_j ] / E_D [ h_hat_i ]

P( h_j = 1 | h_i = 0) = P( h_i = 1, h_j = 0) / P( h_i = 0 )
                      = E_D[ (1-h_hat_i) h_hat_j ] / ( 1- E_D [ h_hat_i ] )
                      = ( E_D[ h_hat_j] - E_D[ h_hat_i h_hat_j] ) / (1 - E_D [ h_hat_i ] )
"""

prod = prod.get_value() / float(batches)
mean = H_mean.get_value() / float(batches)

P1 = (prod.T / mean).T
P0 = ((-prod+mean).T / mean).T

serial.save('prob_diff.pkl',{ 'P0' : P0, 'P1' : P1 } )
