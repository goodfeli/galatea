import numpy as np
from pylearn2.utils import serial
import sys
from dbm_denoise import DBM_Denoise_Binary
import theano.tensor as T
from theano import function
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer

ignore, model_path = sys.argv
model = serial.load(model_path)

try:
    drop_prob = model.dbm_denoise_drop_prob
    n_iter = model.dbm_denoise_n_iter
    balance = model.dbm_inpaint_balance
except:
    drop_prob = 0.5
    n_iter = 5
    balance = False

cost = DBM_Denoise_Binary(drop_prob = drop_prob, n_iter = n_iter,
        balance = balance)

X = T.matrix()

denoising = cost(model,X,return_locals=True)

drop_mask = denoising['drop_mask']
X_hat = denoising['X_hat']

f = function([X],[drop_mask,X_hat])

dataset = yaml_parse.load(model.dataset_yaml_src)

rows = 10
cols = 3
m = rows * cols

X = dataset.get_batch_design(100)

drop_mask, X_hat = f(X)

X, drop_mask, X_hat = [ dataset.adjust_for_viewer(dataset.get_topological_view(mat))
        for mat in [X, drop_mask, X_hat] ]

assert X.shape[-1] == 1 # we assume black and white images that we mark as red in places
pv = PatchViewer( (rows, cols*3), (X.shape[1], X.shape[2]), is_color = True)

for i in xrange(m):
    #add original patch
    patch = X[i,:,:,:]
    patch = np.concatenate( (patch,patch,patch), axis=2)
    pv.add_patch(patch, rescale = False)

    #mark the masked areas as red
    mask_patch = drop_mask[i,:,:,0]
    red_channel = patch[:,:,0]
    other_channels = red_channel.copy()
    red_channel[mask_patch == 1] = 1.
    other_channels[mask_patch == 1] = -1.
    patch[:,:,0] = red_channel
    patch[:,:,1] = other_channels
    patch[:,:,2] = other_channels
    pv.add_patch(patch, rescale = False)

    #add filled-in patch
    patch = X_hat[i,:,:,:]
    patch = np.concatenate( (patch,patch,patch), axis=2)
    pv.add_patch(patch, rescale = False)

pv.show()
