import numpy as np
from pylearn2.utils import serial
import sys
from dbm_inpaint import DBM_Inpaint_Binary
from dbm_inpaint import MaskGen
import theano.tensor as T
from theano import function
from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
from galatea.ui import get_choice
from super_inpaint import SuperInpaint

ignore, model_path = sys.argv
m = 10
model = serial.load(model_path)
if hasattr(model,'set_batch_size'):
    model.set_batch_size(m)

try:
    mask_gen = model.mask_gen
    cost = model.cost
    if not isinstance(cost, (DBM_Inpaint_Binary, SuperInpaint)):
        raise TypeError()
    print 'used cost from model'
    cost.mask_gen = mask_gen
except:
    try:
        drop_prob = model.dbm_inpaint_drop_prob
        n_iter = model.dbm_inpaint_n_iter
        balance = model.dbm_inpaint_balance
    except:
        try:
            drop_prob = model.dbm_denoise_drop_prob
            n_iter = model.dbm_denoise_n_iter
            try:
                balance = model.dbm_denoise_balance
            except:
                balance = False
        except:
            drop_prob = 0.5
            n_iter = 5
            balance = False

    mask_gen = MaskGen(drop_prob = drop_prob, balance = balance)
    from galatea.dbm.inpaint.super_dbm import SuperDBM
    if isinstance(model, SuperDBM):
        from super_inpaint import SuperInpaint
        from super_inpaint import MaskGen
        mask_gen = MaskGen(drop_prob = drop_prob, balance = True, sync_channels = True)
        cost = SuperInpaint(mask_gen = mask_gen)
        print 'made superdbm cost'
    else:
        print 'model is ',type(model)
        cost = DBM_Inpaint_Binary(mask_gen = mask_gen, n_iter = n_iter)
    cost.mask_gen = mask_gen

space = model.get_input_space()
X = space.make_theano_batch()

denoising = cost(model,X,return_locals=True)

drop_mask = denoising['drop_mask']
outputs = [ drop_mask ]
history = denoising['history']
for elem in history:
    try:
        outputs.append(elem['X_hat'])
    except:
        V_hat = elem['V_hat']
        outputs.append(V_hat)


f = function([X],outputs)

print 'use test set?'
choice = get_choice({ 'y' : 'yes', 'n' : 'no' })
if choice == 'y':
    assert model.dataset_yaml_src.find('train') != -1
    model.dataset_yaml_src = model.dataset_yaml_src.replace('train','test')

dataset = yaml_parse.load(model.dataset_yaml_src)

while True:
    if X.ndim == 2:
        X = dataset.get_batch_design(m)
    else:
        X = dataset.get_batch_topo(m)

    outputs = f(X)
    drop_mask = outputs[0]
    print 'empirical drop prob:',drop_mask.mean()
    X_sequence = outputs[1:]


    if X.ndim == 2:
        Xt, drop_mask = [ dataset.get_topological_view(mat)
            for mat in [X, drop_mask] ]
    else:
        Xt = X

    rows = m
    mapback = hasattr(dataset, 'mapback_for_viewer')

    cols = 2+len(X_sequence)
    if mapback:
        rows = 2 * m
        if X.ndim != 2:
            design_X = dataset.get_design_matrix(topo = X)
            design_X_sequence = [ dataset.get_design_matrix(mat) for mat in X_sequence ]
        else:
            design_X = X
            design_X_sequence = X_sequence
        M = dataset.get_topological_view(dataset.mapback_for_viewer(design_X))
        print (M.min(), M.max())
        M_sequence = [ dataset.get_topological_view(dataset.mapback_for_viewer(mat)) for mat in design_X_sequence ]
    X = dataset.adjust_to_be_viewed_with(Xt,Xt,per_example=True)
    if X_sequence[0].ndim == 2:
        X_sequence = [ dataset.get_topological_view(mat) for mat in X_sequence ]
    X_sequence = [ dataset.adjust_to_be_viewed_with(mat,Xt,per_example=True) for mat in X_sequence ]


    pv = PatchViewer( (rows, cols), (X.shape[1], X.shape[2]), is_color = True)

    for i in xrange(m):

        #add original patch
        patch = X[i,:,:,:]
        if patch.shape[-1] != 3:
            patch = np.concatenate( (patch,patch,patch), axis=2)
        pv.add_patch(patch, rescale = False)
        orig_patch = patch

        #mark the masked areas as red
        mask_patch = drop_mask[i,:,:,0]
        if drop_mask.shape[-1] > 1 and mask_gen.n_channels > 1:
            assert np.all(mask_patch == drop_mask[i,:,:,1])
            assert np.all(mask_patch == drop_mask[i,:,:,2])
        red_channel = patch[:,:,0]
        green_channel = patch[:,:,1]
        blue_channel = patch[:,:,2]
        # zeroed patch doesn't handle independent channel masking right, TODO fix that
        zr = red_channel.copy()
        zg = green_channel.copy()
        zb = blue_channel.copy()
        zr[mask_patch == 1] = 0.
        zg[mask_patch == 1] = 0.
        zb[mask_patch == 1] = 0.
        zeroed = patch.copy()
        zeroed[:,:,0] = zr
        zeroed[:,:,1] = zg
        zeroed[:,:,2] = zb
        red_channel[mask_patch == 1] = 1.
        green_channel[mask_patch == 1] = -1.
        blue_channel[mask_patch == 1] = -1.

        if drop_mask.shape[-1] > 1 and mask_gen.n_channels == 1:
            mask_patch = drop_mask[i,:,:,1]
            red_channel[mask_patch == 1] = -1
            green_channel[mask_patch == 1] = 1
            blue_channel[mask_patch == 1] = -1
            mask_patch = drop_mask[i,:,:,2]
            red_channel[mask_patch == 1] = -1
            green_channel[mask_patch == 1] = -1
            blue_channel[mask_patch == 1] = 1

        patch[:,:,0] = red_channel
        patch[:,:,1] = green_channel
        patch[:,:,2] = blue_channel
        pv.add_patch(patch, rescale = False)

        #add filled-in patch
        for j in xrange(len(X_sequence)):
            patch = X_sequence[j][i,:,:,:]
            this_drop_mask = drop_mask[i,:,:,:]
            if ((1-this_drop_mask)*(patch - orig_patch)).max() > 0.:
                keep_mask = 1-this_drop_mask
                keep_patch = keep_mask * patch
                keep_orig = keep_mask* orig_patch
                diffs = keep_patch - keep_orig
                patch = diffs
                print 'OH NO!'
                print patch.shape

            if patch.shape[-1] != 3:
                patch = np.concatenate( (patch,patch,patch), axis=2)
            pv.add_patch(patch, rescale = False)

        if mapback:
            patch = M[i,:,:,:]
            if patch.shape[-1] != 3:
                patch = np.concatenate( (patch,patch,patch),axis=2)
            pv.add_patch(patch, rescale = False)

            # TODO: put this on the right scale
            zeroed = zeroed.reshape( * ( (1,)+zeroed.shape))
            pv.add_patch(dataset.get_topological_view(dataset.mapback(dataset.get_design_matrix(zeroed)))[0,:,:,:], rescale = True)

            #add filled-in patch
            for j in xrange(len(M_sequence)):
                patch = M_sequence[j][i,:,:,:]
                if patch.shape[-1] != 3:
                    patch = np.concatenate( (patch,patch,patch), axis=2)
                pv.add_patch(patch, rescale = False)

    pv.show()

    print 'Waiting...'
    x = raw_input()
    print 'Running...'
