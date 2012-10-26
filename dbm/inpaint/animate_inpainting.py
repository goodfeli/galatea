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

ignore, model_path = sys.argv
m = 10
model = serial.load(model_path)
if hasattr(model,'set_batch_size'):
    model.set_batch_size(m)

try:
    mask_gen = model.mask_gen
    cost = model.cost
    if 'DBM_Inpaint_Binary' not in str(type(cost)) and 'SuperInpaint' not in str(type(cost)):
        if 'SumOfCosts' in str(type(cost)):
            for cost in cost.costs:
                if 'Inpaint' in str(type(cost)):
                    break
        else:
            print type(cost)
            raise TypeError()
    print 'used cost from model'
    cost.mask_gen = mask_gen
except:
    raise
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
if cost.supervised:
    Y = T.matrix()
else:
    Y = None

print 'cost is',str(cost)
denoising = cost(model,X,Y,return_locals=True)

drop_mask = denoising['drop_mask']
outputs = [ drop_mask ]
history = denoising['history']
for elem in history:
    try:
        outputs.append(elem['X_hat'])
    except:
        V_hat = elem['V_hat']
        outputs.append(V_hat)
end_X_outputs = len(outputs)

inputs = [X]
if cost.supervised:
    inputs += [Y]
    outputs += [ denoising['drop_mask_Y'] ]
    for elem in history:
        outputs.append(elem['Y_hat'])

f = function(inputs, outputs)

dataset = yaml_parse.load(model.dataset_yaml_src)

if cost.supervised:
    n_classes = model.hidden_layers[-1].n_classes
    if isinstance(n_classes, float):
        assert n_classes == int(n_classes)
        n_classes = int(n_classes)
    assert isinstance(n_classes, int)
    templates = np.zeros((n_classes, space.get_total_dimension()))
    for i in xrange(n_classes):
        for j in xrange(-1, -dataset.X.shape[0], -1):
            if dataset.y[j,i]:
                templates[i, :] = dataset.X[j, :]

print 'use test set?'
choice = get_choice({ 'y' : 'yes', 'n' : 'no' })
if choice == 'y':
    dataset = dataset.get_test_set()



topo = X.ndim > 2

while True:
    if cost.supervised:
        X, Y = dataset.get_batch_design(m, include_labels = True)
    else:
        X = dataset.get_batch_design(m)
    if topo:
        X = dataset.get_topological_view(X)

    args = [X]
    if cost.supervised:
        args += [Y]

    outputs = f(*args)
    drop_mask = outputs[0]
    print 'empirical drop prob:',drop_mask.mean()
    X_sequence = outputs[1:end_X_outputs]


    if X.ndim == 2:
        Xt, drop_mask = [ dataset.get_topological_view(mat)
            for mat in [X, drop_mask] ]
    else:
        Xt = X

    rows = m
    mapback = hasattr(dataset, 'mapback_for_viewer')

    cols = 2+len(X_sequence)
    if mapback:
        rows += m
        if X.ndim != 2:
            design_X = dataset.get_design_matrix(topo = X)
            design_X_sequence = [ dataset.get_design_matrix(mat) for mat in X_sequence ]
        else:
            design_X = X
            design_X_sequence = X_sequence
        zeroed = (1.-drop_mask) * X
        zeroed = dataset.get_topological_view(dataset.mapback_for_viewer(dataset.get_design_matrix(zeroed)))
        M = dataset.get_topological_view(dataset.mapback_for_viewer(design_X))
        print (M.min(), M.max())
        M_sequence = [ dataset.get_topological_view(dataset.mapback_for_viewer(mat)) for mat in design_X_sequence ]
    X = dataset.adjust_to_be_viewed_with(Xt,Xt,per_example=True)
    if X_sequence[0].ndim == 2:
        X_sequence = [ dataset.get_topological_view(mat) for mat in X_sequence ]
    X_sequence = [ dataset.adjust_to_be_viewed_with(mat,Xt,per_example=True) for mat in X_sequence ]

    if cost.supervised:
        rows += m

        drop_mask_Y = outputs[end_X_outputs]
        Y_sequence = outputs[end_X_outputs+1:]


    pv = PatchViewer( (rows, cols), (X.shape[1], X.shape[2]), is_color = True,
            pad = (8,8) )

    for i in xrange(m):

        #add original patch
        patch = X[i,:,:,:]
        if patch.shape[-1] != 3:
            patch = np.concatenate( (patch,patch,patch), axis=2)
        pv.add_patch(patch, rescale = False, activation = (1,0,0))
        orig_patch = patch

        #mark the masked areas as red
        mask_patch = drop_mask[i,:,:,0]
        if drop_mask.shape[-1] > 1 and mask_gen.sync_channels and mask_gen.n_channels > 1:
            assert np.all(mask_patch == drop_mask[i,:,:,1])
            assert np.all(mask_patch == drop_mask[i,:,:,2])
        red_channel = patch[:,:,0]
        green_channel = patch[:,:,1]
        blue_channel = patch[:,:,2]
        mask_patch_red = drop_mask[i,:,:,0]
        if mask_gen.sync_channels or drop_mask.shape[-1] == 1:
            mask_patch_green = mask_patch_red
            mask_patch_blue = mask_patch_red
        else:
            mask_patch_green = drop_mask[i,:,:,1]
            mask_patch_blue = drop_mask[i,:,:,2]
        zr = red_channel.copy()
        zg = green_channel.copy()
        zb = blue_channel.copy()
        zr[mask_patch_red == 1] = 0.
        zg[mask_patch_green == 1] = 0.
        zb[mask_patch_blue == 1] = 0.
        red_channel[mask_patch == 1] = 1.
        green_channel[mask_patch == 1] = -1.
        blue_channel[mask_patch == 1] = -1.

        if drop_mask.shape[-1] > 1:
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
        pv.add_patch(patch, rescale = False, activation = (1,0,0))

        #add filled-in patch
        for j in xrange(len(X_sequence)):
            patch = X_sequence[j][i,:,:,:]
            this_drop_mask = drop_mask[i,:,:,:]
            if mask_gen.sync_channels and ((1-this_drop_mask)*(patch - orig_patch)).max() > 0.:
                keep_mask = 1-this_drop_mask
                keep_patch = keep_mask * patch
                keep_orig = keep_mask* orig_patch
                diffs = keep_patch - keep_orig
                patch = diffs
                print 'OH NO!'
                print patch.shape

            if patch.shape[-1] != 3:
                patch = np.concatenate( (patch,patch,patch), axis=2)
            pv.add_patch(patch, rescale = False, activation = (1,0,0))

        if mapback:
            patch = M[i,:,:,:]
            if patch.shape[-1] != 3:
                patch = np.concatenate( (patch,patch,patch),axis=2)
            pv.add_patch(patch, rescale = False, activation= (0,1,0))

            patch = zeroed[i,:,:,:]
            pv.add_patch(patch, rescale = True, activation = (0,1,0))

            #add filled-in patch
            for j in xrange(len(M_sequence)):
                patch = M_sequence[j][i,:,:,:]
                if patch.shape[-1] != 3:
                    patch = np.concatenate( (patch,patch,patch), axis=2)
                pv.add_patch(patch, rescale = False, activation=(0,1,0))

        if cost.supervised:
            def label_to_vis(Y_elem):
                prod =  np.dot(Y_elem, templates)
                assert Y_elem.ndim == 1
                rval = np.zeros((1, prod.shape[0]))
                rval[0,:] = prod
                return rval
            #Show true class
            Y_vis = label_to_vis(Y[i,:])
            Y_vis = dataset.adjust_for_viewer(dataset.get_topological_view(Y_vis))
            if Y_vis.ndim == 2:
                Y_vis = Y_vis.reshape(Y_vis.shape[0], Y_vis.shape[1], 1)
            if Y_vis.ndim == 4:
                assert Y_vis.shape[0] == 1
                Y_vis = Y_vis[0,:,:,:]
            assert Y_vis.ndim == 3
            if Y_vis.shape[-1] == 1:
                Y_vis = np.concatenate([Y_vis]*3,axis=2)
            assert Y_vis.shape[-1] == 3
            pv.add_patch(Y_vis, rescale=False, activation=(0,0,1))

            # Add the masked input
            if drop_mask_Y[i]:
                pv.add_patch(np.concatenate((np.ones((X.shape[1], X.shape[2], 1)),
                                            -np.ones((X.shape[1], X.shape[2], 1)),
                                            -np.ones((X.shape[1], X.shape[2], 1))),
                                            axis=2), rescale = False,
                                            activation =(0,0,1))
            else:
                pv.add_patch(Y_vis, activation=(0,0,1))

            # Add the inpainted sequence
            for Y_hat in Y_sequence:
                cur_Y_hat = Y_hat[i,:]
                Y_vis = label_to_vis(cur_Y_hat)
                Y_vis = dataset.adjust_for_viewer(dataset.get_topological_view(Y_vis))
                if Y_vis.ndim == 4:
                    assert Y_vis.shape[0] == 1
                    Y_vis = Y_vis[0,:,:,:]
                if Y_vis.ndim == 2:
                    Y_vis = Y_vis.reshape(Y_vis.shape[0], Y_vis.shape[1], 1)
                if Y_vis.shape[-1] == 1:
                    Y_vis = np.concatenate([Y_vis]*3,axis=2)
                pv.add_patch(Y_vis, rescale=False, activation=(0,0,1))


    pv.show()

    print 'Waiting...'
    x = raw_input()
    if x == 'q':
        break
    print 'Running...'
