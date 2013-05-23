import numpy as np
import warnings

from pylearn2.utils import serial
import sys
import theano.tensor as T
from theano import function

from pylearn2.config import yaml_parse
from pylearn2.gui.patch_viewer import PatchViewer
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX

from galatea.ui import get_choice
from pylearn2.models.dbm import flatten
from theano.sandbox.rng_mrg import MRG_RandomStreams

from galatea.dbm.inpaint.super_inpaint import MaskGen

ignore, model_path = sys.argv
m = 10
model = serial.load(model_path)
if hasattr(model,'set_batch_size'):
    model.set_batch_size(m)

dataset = yaml_parse.load(model.dataset_yaml_src)

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
        mask_gen = MaskGen(drop_prob = drop_prob, balance = True, sync_channels
                = dataset.get_batch_topo(1).shape[-1] > 1)
        cost = SuperInpaint(mask_gen = mask_gen)
        print 'made superdbm cost'
    else:
        print 'model is ',type(model)
        cost = DBM_Inpaint_Binary(mask_gen = mask_gen, n_iter = n_iter)
    cost.mask_gen = mask_gen


space = model.get_input_space()
X = space.make_theano_batch()
x = raw_input("mask half of image?")
if x == 'y':
    if X.ndim == 4:
            class DummyMaskGen(object):
                sync_channels = 0

                def __call__(self, X, Y=None, X_space=None):
                    if tuple(X_space.axes) != ('b', 0, 1, 'c'):
                        print X_space.axes
                        raise NotImplementedError()
                    left_mask = T.ones_like(X[:, :, 0:X.shape[2]/2, :])
                    right_mask = T.zeros_like(X[:, :, X.shape[2]/2:, :])
                    X_mask = T.concatenate((left_mask, right_mask), axis=2)
                    Y_mask = T.zeros_like(Y[:,0])
                    return X_mask, Y_mask
    else:
        assert X.ndim == 2
        mask = dataset.get_batch_topo(m)
        mask *= 0.
        mask[:, :, 0:mask.shape[2] /2, :] = 1
        if hasattr(dataset, 'raw'):
            mask = dataset.raw.get_design_matrix(mask)
        else:
            mask = dataset.get_design_matrix(mask)
        mask = sharedX(mask)

        class DummyMaskGen(object):
            sync_channels = 0

            def __call__(self, X, Y=None):
                if Y is None:
                    return mask

                x = raw_input('mask out y?')

                if x == 'y':
                    return mask, T.ones_like(Y[:,0])
                assert x == 'n'
                return mask, T.zeros_like(Y[:,0])
    mask_gen = DummyMaskGen()

    cost.mask_gen = mask_gen

    x = raw_input('override number mf iterations? (# or n) ')
    if x != 'n':
        model.niter = int(x)



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

topo = X.ndim > 2

if cost.supervised:
    n_classes = model.hidden_layers[-1].n_classes
    if isinstance(n_classes, float):
        assert n_classes == int(n_classes)
        n_classes = int(n_classes)
    assert isinstance(n_classes, int)
    templates = np.zeros((n_classes, space.get_total_dimension()))
    X, y = dataset.get_batch_design(5000, include_labels=1)
    for i in xrange(n_classes):
        for j in xrange(-1, -y.shape[0], -1):
            if y[j,i]:
                templates[i, :] = X[j, :]

print 'use test set?'
choice = get_choice({ 'y' : 'yes', 'n' : 'no' })
if choice == 'y':
    dataset = dataset.get_test_set()


dropout = hasattr(model.inference_procedure, 'V_dropout')
if dropout:
    include_prob = model.inference_procedure.include_prob
    theano_rng = MRG_RandomStreams(2012+11+20)
    updates = {}
    for elem in flatten([model.inference_procedure.V_dropout, model.inference_procedure.H_dropout]):
        updates[elem] =  theano_rng.binomial(p=include_prob, size=elem.shape, dtype=elem.dtype, n=1) / include_prob
    do_dropout = function([], updates=updates)


while True:
    if dropout:
        do_dropout()

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


    if topo:
        Xt = X

    else:
        Xt, drop_mask = [ dataset.get_topological_view(mat)
            for mat in [X, drop_mask] ]

    rows = m
    mapback = hasattr(dataset, 'mapback_for_viewer')

    if mapback and model.get_input_space().axes != ('b', 0, 1, 'c'):
        warnings.warn("Disabling mapback, not implemented for general axes yet")
        mapback = False

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
    else:
        model_axes = model.get_input_space().axes
        display_axes = ('b', 0, 1, 'c')

        X = Conv2DSpace.convert_numpy(X, model_axes, display_axes)
        Xt = Conv2DSpace.convert_numpy(Xt, model_axes, display_axes)
        drop_mask = Conv2DSpace.convert_numpy(drop_mask, model_axes, display_axes)
        X_sequence = [Conv2DSpace.convert_numpy(elem, model_axes, display_axes) for elem in X_sequence]
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
            assert patch.shape[-1] == 2, patch.shape
            patch = patch[:,:,0:1]
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
        if drop_mask.shape[-1] == 2:
            drop_mask = drop_mask[:, :, :, 0:1]
            assert drop_mask.shape[-1] == 1, drop_mask.shape
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
                if patch.shape[-1] == 2:
                    patch = patch[:, :, 0:1]
                    assert patch.shape[-1] == 1
                patch = np.concatenate( (patch,patch,patch), axis=2)
                assert patch.shape[-1] == 3
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
            Y_vis = np.clip(label_to_vis(Y[i,:]), -1., 1.)
            Y_topo = dataset.get_topological_view(Y_vis)
            if 'model_axes' in locals():
                Y_topo = Conv2DSpace.convert_numpy(Y_topo, model_axes, display_axes)
            Y_vis = dataset.adjust_for_viewer(Y_topo)
            if Y_vis.ndim == 2:
                Y_vis = Y_vis.reshape(Y_vis.shape[0], Y_vis.shape[1], 1)
            if Y_vis.ndim == 4:
                assert Y_vis.shape[0] == 1
                Y_vis = Y_vis[0,:,:,:]
            assert Y_vis.ndim == 3
            if Y_vis.shape[-1] == 2:
                Y_vis = Y_vis[:,:, 0:1]
                assert Y_vis.shape[-1] == 1
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
                Y_topo = dataset.get_topological_view(Y_vis)
                if 'model_axes' in locals():
                    Y_topo = Conv2DSpace.convert_numpy(Y_topo, model_axes, display_axes)
                Y_vis = dataset.adjust_for_viewer(Y_topo)
                if Y_vis.ndim == 4:
                    assert Y_vis.shape[0] == 1
                    Y_vis = Y_vis[0,:,:,:]
                if Y_vis.ndim == 2:
                    Y_vis = Y_vis.reshape(Y_vis.shape[0], Y_vis.shape[1], 1)
                if Y_vis.shape[-1] == 2:
                    Y_vis = Y_vis[:,:,0:1]
                if Y_vis.shape[-1] == 1:
                    Y_vis = np.concatenate([Y_vis]*3,axis=2)
                pv.add_patch(np.clip(Y_vis, -1., 1.), rescale=False, activation=(0,0,1))


    pv.show()

    print 'Waiting...'
    x = raw_input()
    if x == 'q':
        break
    print 'Running...'
