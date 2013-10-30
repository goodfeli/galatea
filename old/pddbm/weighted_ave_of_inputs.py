import sys

ignore, model_path = sys.argv

from pylearn2.utils import serial
from pylearn2.utils import sharedX
import theano.tensor as T

print 'loading model'
model = serial.load(model_path)

print 'building theano graph'
IS = model.get_input_space()

V = IS.make_theano_batch()

ave_V_s = sharedX(IS.get_origin_batch(model.s3c.nhid))
ave_V_h = sharedX(IS.get_origin_batch(model.s3c.nhid))

assert len(model.dbm.rbms) == 1

ave_V_g = sharedX(IS.get_origin_batch(model.dbm.rbms[0].nhid))

ip = model.inference_procedure

obs = ip.infer(V)

from theano import function

S = obs['S_hat']
H = obs['H_hat']
G = obs['G_hat'][0]

print 'compiling function'
f = function([V], updates = {
        ave_V_s : ave_V_s + T.dot(S.T,V),
        ave_V_h : ave_V_h + T.dot(H.T,V),
        ave_V_g : ave_V_g + T.dot(G.T,V),
        })


print 'loading dataset'
from pylearn2.config import yaml_parse
dataset = yaml_parse.load(model.dataset_yaml_src)

batch_size = 100
batches = 50

for i in xrange(batches):
    print 'batch ',i
    X = dataset.get_batch_design(batch_size)

    f(X)

H = ave_V_h.get_value()
S = H * ave_V_s.get_value()
G = ave_V_g.get_value()


from pylearn2.gui.patch_viewer import make_viewer

pv1 = make_viewer(S)
pv1.show()
pv2 = make_viewer(H)
pv2.show()
pv3 = make_viewer(G)
pv3.show()



