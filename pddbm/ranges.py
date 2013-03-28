import sys
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
import theano.tensor as T
import numpy as np

batch_size = 100

ignore, model_path = sys.argv

print 'loading model...'
model = serial.load(model_path)
model.make_pseudoparams()

print 'loading dataset...'
dataset  = yaml_parse.load(model.dataset_yaml_src)

print 'compiling function'
V = T.matrix()
obs = model.inference_procedure.infer(V)

assert obs['H_hat'].ndim == 2

G_max = T.max(obs['G_hat'][0],axis=0)
G_min = T.min(obs['G_hat'][0],axis=0)
H_max = T.max(obs['H_hat'],axis=0)
assert H_max.ndim == 1
H_min = T.min(obs['H_hat'],axis=0)
assert H_min.ndim == 1

f = function([V],[H_min,H_max,G_min,G_max])

#set up tracker vars
G_max = np.zeros((model.dbm.rbms[0].nhid,))
G_min = np.ones(G_max.shape)
H_max = np.zeros((model.s3c.nhid,))
H_min = np.ones(H_max.shape)

i = 1
while True:
    print 'processing batch',i
    X = dataset.get_batch_design(batch_size)
    cur_H_min, cur_H_max, cur_G_min, cur_G_max = f(X)
    assert len(cur_H_min.shape) == 1
    assert len(cur_H_max.shape) == 1
    H_min = np.minimum(H_min, cur_H_min)
    H_max = np.maximum(H_max, cur_H_max)
    G_min = np.minimum(G_min, cur_G_min)
    G_max = np.maximum(G_max, cur_G_max)
    h_range = H_max - H_min
    g_range = G_max - G_min
    print 'h range: ',(h_range.min(),h_range.mean(),h_range.max())
    print 'g range: ',(g_range.min(),g_range.mean(),g_range.max())
    i = i + 1
