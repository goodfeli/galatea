import numpy as np

import sys
model_path = sys.argv[1]

indiv_rescale = True
pair_rescale = True
global_rescale = False

from pylearn2.utils import serial
print 'loading model...'
model = serial.load(model_path)
print 'done'

from pylearn2.config import yaml_parse
print 'loading dataset...'
dataset = yaml_parse.load(model.dataset_yaml_src)
print 'done'

import theano.tensor as T
from theano import function
if hasattr(model,'make_pseudoparams'):
    model.make_pseudoparams()

def get_reconstruction_func():
    V = T.matrix()

    if hasattr(model,'e_step'):
        #S3C
        mf = model.e_step.variational_inference(V)
        H = mf['H_hat']
        S = mf['S_hat']
        Z = H*S
        recons = T.dot(Z,model.W.T)
    elif hasattr(model,'s3c'):
        #PDDBM
        ip = model.inference_procedure
        mf = ip.infer(V)

        x = raw_input('reconstruct from which layer? ')

        if x == '1':
            H = mf['H_hat']
            x = raw_input('use S from inference (y = yes, n = use mu) ? ')
            if x == 'y':
                S = mf['S_hat']
            elif x == 'n':
                S = ip.s3c_e_step.init_S_hat(V)
            else:
                assert False
            Z = H*S
            recons = T.dot(Z,model.s3c.W.T)
        elif x == '2':
            G1 = mf['G_hat'][0]

            H = ip.dbm_ip.infer_H_hat_one_sided(other_H_hat = G1,
                    W = model.dbm.W[0].T,
                    b = model.dbm.bias_vis)

            x = raw_input('use S from inference (y = yes, n = use mu) ? ')
            if x == 'y':
                S = mf['S_hat']
            elif x == 'n':
                S = ip.s3c_e_step.init_S_hat(V)
            else:
                assert False
            Z = H*S

            recons = T.dot(Z,model.s3c.W.T)

            #recons = G1
        else:
            raise NotImplementedError()
    else:
        #RBM
        H = model.mean_h_given_v(V)
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        theano_rng = RandomStreams(42)
        H_sample = theano_rng.binomial(size = H.shape, p = H)
        from theano.printing import Print
        H_sample = Print('H_sample', attrs=['mean'])(H_sample)
        recons = model.mean_v_given_h(H_sample)
        recons = Print('recons', attrs=['min','mean','max'])(recons)


    rval = function([V],recons)

    return rval

print 'making reconstruction function...'
f = get_reconstruction_func()
print 'done'

if hasattr(model, 'random_patches_src'):
    dataset.get_batch_design(model.nhid)

n = 50
if hasattr(dataset, 'get_unprocessed_batch_design'):
    Xr = dataset.get_unprocessed_batch_design(50)
    Xt = dataset.raw.get_topological_view(Xr)
    X = dataset.transformer.perform(Xr)
else:
    X = dataset.get_batch_design(50)
    Xt = dataset.get_topological_view(X)

R = f(X)

#mn = R.mean(axis=0)
#d = R - mn
#d = np.abs(d)
#d = d.mean(axis=0)
#print (d.min(),d.mean(),d.max())
#assert False


if np.any(np.isnan(R)) or np.any(np.isinf(R)):
    mask = (np.isnan(R).sum(axis=1) + np.isinf(R).sum(axis=1)) > 0
    print float(mask.sum())/float(mask.shape[0])
    print R
    assert False

print 'mean squared error: ',np.square(X-R).mean()

Rt = dataset.get_topological_view(R)

from pylearn2.gui.patch_viewer import PatchViewer

num_patches = X.shape[0] * 2

import numpy as np

n = int( np.sqrt( num_patches ) )

rows = n
cols = n

if cols % 2 != 0:
    cols += 1

if rows * cols < num_patches:
    rows += 1

pv = PatchViewer( grid_shape = (rows, cols), patch_shape = Xt.shape[1:3], is_color = Xt.shape[3] > 1)

if global_rescale:
    scale = max(np.abs(Xt).max(),np.abs(Rt).max())
    Xt /= scale
    Rt /= scale
    scale = 1.

for i in xrange(X.shape[0]):
    x = Xt[i,:]
    r = Rt[i,:]

    if pair_rescale:
        scale = max(np.abs(x).max(),np.abs(r).max())

    assert scale != 0.0

    pv.add_patch( x / scale, rescale = indiv_rescale, activation = 0)
    pv.add_patch( r / scale, rescale = indiv_rescale, activation = 0)

pv.show()
