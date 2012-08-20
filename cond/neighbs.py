from theano.sandbox.neighbours import images2neibs
from theano.tensor import as_tensor_variable
from theano import tensor as T

def cifar10neighbs(topo, patch_shape):
    r, c = patch_shape
    topo = as_tensor_variable(topo)
    flat =  images2neibs(ten4 = topo.dimshuffle(3,0,1,2),
            neib_shape = (r,c),
            neib_step = (1,1))
    m = flat.shape[0] / 3
    n = flat.shape[1] * 3
    rval = flat.reshape(m,n)
    #red = flat[0:m,:]
    #green = flat[m:2*m,:]
    #blue = flat[2*m:,:]

    #rval = T.concatenate((red,green,blue),axis=1)
    return rval

