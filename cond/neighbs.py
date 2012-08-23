from theano.sandbox.neighbours import images2neibs
from theano.sandbox.neighbours import neibs2images
from theano.tensor import as_tensor_variable
from theano import tensor as T
#test by running show_map and show_neighbs

def cifar10neighbs(topo, patch_shape):
    assert topo.ndim == 4
    r, c = patch_shape
    topo = as_tensor_variable(topo)
    flat =  images2neibs(ten4 = topo.dimshuffle(3,0,1,2),
            neib_shape = (r,c),
            neib_step = (1,1))
    m = flat.shape[0] / 3
    n = flat.shape[1] * 3

    red = flat[0:m,:]
    green = flat[m:2*m,:]
    blue = flat[2*m:,:]

    rval = T.concatenate((red,green,blue),axis=1)
    return rval


def multichannel_neibs2imgs(X, batch_size, map_rows, map_cols, channels, neib_rows, neib_cols):
    X = as_tensor_variable(X)
    convmap = X.reshape((batch_size, map_rows, map_cols, channels * neib_rows * neib_cols))
    return convmap
