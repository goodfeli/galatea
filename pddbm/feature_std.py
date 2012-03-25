batch_size = 100

import sys

ignore, model_path = sys.argv

from pylearn2.utils import serial

model = serial.load(model_path)

from pylearn2.config import yaml_parse

dataset = yaml_parse.load(model.dataset_yaml_src)

X = dataset.get_batch_design(batch_size)

from theano import tensor as T

V = T.matrix()

if hasattr(model,'inference_procedure'):
    ip = model.inference_procedure
    obs = ip.infer(V)
else:
    model.make_pseudoparams()
    ip = model.e_step
    obs = ip.variational_inference(V)

outputs = [ obs['S_hat'], obs['H_hat'] ]

if 'G_hat' in obs:
    for G in obs['G_hat']:
        outputs.append(G)

from theano import function

f = function([V],outputs)

outputs = f(X)


print 'std'
for output in outputs:
    std = output.std(axis=0)
    print (std.min(),std.mean(),std.max())
print 'rng'
for output in outputs:
    rng = output.max(axis=0) - output.min(axis=0)
    print (rng.min(),rng.mean(),rng.max())
