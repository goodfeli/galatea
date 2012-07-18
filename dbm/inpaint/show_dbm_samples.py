import sys
import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import theano.tensor as T
from theano import function
from pylearn2.gui.patch_viewer import make_viewer
import numpy as np

ignore, model_path = sys.argv

model = serial.load(model_path)
dbm = model
dataset = yaml_parse.load(model.dataset_yaml_src)

theano_rng = RandomStreams(42)

X = T.matrix()
X.tag.test_value = np.zeros((2,dbm.rbms[0].nvis),dtype=X.dtype)
G = T.matrix()
G.tag.test_value = np.zeros((2,dbm.rbms[1].nhid),dtype=G.dtype)
H = T.matrix()
H.tag.test_value = np.zeros((2,dbm.rbms[0].nhid),dtype=G.dtype)
ip = dbm.inference_procedure
Hprime = theano_rng.binomial(
        size = H.shape,
        n = 1,
        dtype = H.dtype,
        p = ip.infer_H_hat_two_sided(
            H_hat_below = X,
            H_hat_above = G,
            W_below = dbm.W[0],
            W_above = dbm.W[1],
            b = dbm.bias_hid[0]))
Gprime = theano_rng.binomial(
        size = G.shape,
        n = 1,
        dtype = G.dtype,
        p = ip.infer_H_hat_one_sided(
            other_H_hat = Hprime,
            W = dbm.W[1],
            b = dbm.bias_hid[1]))
Xprime = theano_rng.binomial(
        size = X.shape,
        n = 1,
        dtype = X.dtype,
        p = ip.infer_H_hat_one_sided(
            other_H_hat = Hprime,
            W = dbm.W[0].T,
            b = dbm.bias_vis))

f = function([G,H,X],[Gprime,Hprime,Xprime])

if ip.layer_schedule is None:
    ip.layer_schedule = [0,1] * 10
obs = ip.infer(X)
pH, pG = obs['H_hat']
sample_from_posterior = function([X],
    [ theano_rng.binomial( size = pG.shape,
        p = pG, n = 1, dtype = pG.dtype),
        theano_rng.binomial( size = pH.shape,
            p = pH, n = 1, dtype = pH.dtype) ] )


m = 100
X = dataset.get_batch_design(m)
G, H = sample_from_posterior(X)
assert len(dbm.rbms) == 2

while True:
    V = dataset.adjust_for_viewer(X)
    viewer = make_viewer(V, is_color = X.shape[1] % 3 == 0)
    viewer.show()

    print 'Waiting...'
    x = raw_input()
    if x == 'q':
        break
    print 'Running...'

    num_updates = 1

    try:
        num_updates = int(x)
    except:
        pass

    for i in xrange(num_updates):
        G,H,X = f(G,H,X)

