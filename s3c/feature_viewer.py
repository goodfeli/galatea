import numpy as np
from pylearn2.datasets.cifar10 import CIFAR10
from theano import config
from theano import tensor as T
from theano import function
from optparse import OptionParser
from pylearn2.datasets.preprocessing import ExtractPatches, ExtractGridPatches, ReassembleGridPatches
from pylearn2.gui.patch_viewer import make_viewer
from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.gui.patch_viewer import PatchViewer
import warnings
if config.floatX != 'float32':
    warnings.warn('setting floatX to float32 to mimic feature extractor')
config.floatX = 'float32'

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-m", "--model",
                action="store", type="string", dest="model")
    parser.add_option("-s","--batch-start", action="store", dest="batch_start", type="int", default=0,
                      help="the index of the start of the batch of examples to use")
    parser.add_option("-b","--batch-size", action="store", dest="batch_size", type="int", default=15,
                      help="")
    parser.add_option("-f","--filter-start", action="store", dest="filter_start", type="int", default=0,
                      help="the index of the start of the range of fitlers to display")
    parser.add_option("-n","--num-filters", action="store", dest="num_filters", type="int", default=15,
                      help="how many filters to show")
    parser.add_option("-t","--feature-type", action="store", dest="feature_type", type="string", default='exp_hs',
            help="""exp_hs (default): expectation of hs
exp_h: expectation of h
map_hs: h*s, computed from joint MAP estimation under q
""")

    (options, args) = parser.parse_args()

    if options.model is None:
        raise ValueError('The -m/--model flag is required')

    batch_start = options.batch_start
    batch_size = options.batch_size
    filter_start = options.filter_start
    num_filters = options.num_filters
    feature_type = options.feature_type

    assert feature_type in ['exp_hs','exp_h','map_hs']

    model_path = options.model

    print 'loading model'
    model = serial.load(model_path)

    stl10 = False
    cifar10 = False
    stl10 = model.dataset_yaml_src.find('stl10') != -1
    cifar10 = model.dataset_yaml_src.find('cifar10') != -1
    assert stl10 or cifar10
    assert not (stl10 and cifar10)

    print 'loading dataset'
    if cifar10:
        print 'CIFAR10 detected'
        dataset = CIFAR10(which_set = "train")
    elif stl10:
        print 'STL10 detected'
        dataset = serial.load('${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/train.pkl')
    X = dataset.get_design_matrix()[batch_start:batch_start + batch_size,:]

    size = np.sqrt(model.nvis/3)

    if cifar10:
        pv1 = make_viewer( (X-127.5)/127.5, is_color = True, rescale = False)
    elif stl10:
        pv1 = make_viewer( X/127.5, is_color = True, rescale = False)

    dataset.set_design_matrix(X)

    patchifier = ExtractGridPatches( patch_shape = (size,size), patch_stride = (1,1) )


    if size == 8:
        if cifar10:
            pipeline = serial.load('${GOODFELI_TMP}/cifar10_preprocessed_pipeline_2M.pkl')
        elif stl10:
            assert False
    elif size ==6:
        if cifar10:
            pipeline = serial.load('${GOODFELI_TMP}/cifar10_preprocessed_pipeline_2M_6x6.pkl')
        elif stl10:
            pipeline = serial.load('${PYLEARN2_DATA_PATH}/stl10/stl10_patches/preprocessor.pkl')
    else:
        print size
        assert False

    assert isinstance(pipeline.items[0], ExtractPatches)
    pipeline.items[0] = patchifier

    print 'applying preprocessor'
    dataset.apply_preprocessor(pipeline, can_fit = False)


    X2 = dataset.get_design_matrix()

    print 'defining features'
    V = T.matrix()
    model.make_Bwp()
    d = model.e_step.mean_field(V = V)

    H = d['H']
    Mu1 = d['Mu1']

    if feature_type == 'exp_hs':
        feat = H * Mu1
    elif feature_type == 'exp_h':
        feat = H
    elif feature_type == 'map_hs':
        feat = ( H > 0.5) * Mu1
    else:
        assert False

    print 'compiling theano function'
    f = function([V],feat)

    print 'running theano function'
    feat = f(X2)

    feat_dataset = DenseDesignMatrix(X = feat, view_converter = DefaultViewConverter([1, 1, feat.shape[1]] ) )

    print 'reassembling features'
    ns = 32 - size + 1
    depatchifier = ReassembleGridPatches( orig_shape  = (ns, ns), patch_shape=(1,1) )
    feat_dataset.apply_preprocessor(depatchifier)

    print 'making topological view'
    topo_feat = feat_dataset.get_topological_view()
    assert topo_feat.shape[0] == X.shape[0]

    print 'assembling visualizer'

    n = np.ceil(np.sqrt(model.nhid))

    pv3 = PatchViewer(grid_shape = (X.shape[0], num_filters), patch_shape=(ns,ns), is_color= False)
    pv4 = PatchViewer(grid_shape = (n,n), patch_shape = (size,size), is_color = True, pad = (7,7))
    pv5 = PatchViewer(grid_shape = (1,num_filters), patch_shape = (size,size), is_color = True, pad = (7,7))

    idx = sorted(range(model.nhid), key = lambda l : -topo_feat[:,:,:,l].std() )

    W = model.W.get_value()

    weights_view = dataset.get_weights_view( W.T )

    p_act = 1. / (1. + np.exp(- model.bias_hid.get_value()))
    p_act /= p_act.max()

    mu_act = np.abs(model.mu.get_value())
    mu_act /= mu_act.max()
    mu_act += 0.5

    alpha_act = model.alpha.get_value()
    alpha_act /= alpha_act.max()

    for j in xrange(model.nhid):
        cur_idx = idx[j]

        cur_p_act = p_act[cur_idx]
        cur_mu_act = mu_act[cur_idx]
        cur_alpha_act = alpha_act[cur_idx]

        activation = (cur_p_act, cur_mu_act, cur_alpha_act)

        pv4.add_patch(weights_view[cur_idx,:], rescale = True,
                activation = activation)

        if j >= filter_start and j < filter_start + num_filters:
            pv5.add_patch(weights_view[cur_idx,:], rescale = True,
                activation = activation)


    for i in xrange(X.shape[0]):
        #indices of channels sorted in order of descending standard deviation on this example
        #plot the k most interesting channels
        for j in xrange(filter_start, filter_start+num_filters):
            pv3.add_patch(topo_feat[i,:,:,idx[j]], rescale = True, activation = 0.)

    pv1.show()
    pv3.show()
    pv4.show()
    pv5.show()
