import theano.tensor as T
import numpy as np
from theano import config
from theano import function
import time
from pylearn2.utils import sharedX

def max_pool_python(z, pool_shape):

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    assert zr % r == 0
    assert zc % c == 0

    h = np.zeros(z.shape, dtype = z.dtype)
    p = np.zeros( (batch_size, zr /r, zc /c, ch), dtype = z.dtype)

    for u in xrange(0,zr,r):
        for l in xrange(0,zc,c):
            pt = np.exp(z[:,u:u+r,l:l+c,:])
            denom = pt.sum(axis=1).sum(axis=1) + 1.
            p[:,u/r,l/c,:] = 1. - 1. / denom
            for i in xrange(batch_size):
                for j in xrange(ch):
                    pt[i,:,:,j] /= denom[i,j]
            h[:,u:u+r,l:l+c,:] = pt

    return p, h


def max_pool_raw(z, pool_shape):
    #random max pooling implemented with set_subtensor
    #could also do this using the stuff in theano.sandbox.neighbours
    #might want to benchmark the two approaches, see how each does on speed/memory
    #on cpu and gpu
    #this method is not numerically stable, use max_pool instead

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    zpart = []

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            zpart[i].append( z[:,i:zr:r,j:zc:c,:] )

    pt = []

    for i in xrange(r):
        pt.append( [ T.exp(z_ij) for z_ij in zpart[i] ] )

    denom = 1.

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]

    p = 1. - 1. / denom

    hpart = []
    for i in xrange(r):
        hpart.append( [ pt_ij / denom for pt_ij in pt[i] ] )

    h = T.alloc(0., batch_size, zr, zc, ch)

    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:,i:zr:r,j:zc:c,:],hpart[i][j])

    return p, h

def max_pool(z, pool_shape):
    #random max pooling implemented with set_subtensor
    #could also do this using the stuff in theano.sandbox.neighbours
    #might want to benchmark the two approaches, see how each does on speed/memory
    #on cpu and gpu
    #here I stabilized the softplus with 4 calls to T.maximum and 5 elemwise
    #subs. this is 10% slower than the unstable version, and the gradient
    #is 40% slower. on GPU both the forward prop and backprop are more like
    #100% slower!
    #might want to dry doing a reshape, a T.nnet.softplus, and a reshape
    #instead
    #another way to implement the stabilization is with the max pooling operator
    #(you'd still need to do maximum with 0)


    #timing hack
    #return T.nnet.sigmoid(z[:,0:z.shape[1]/pool_shape[0],0:z.shape[2]/pool_shape[1],:]), T.nnet.sigmoid(z)

    z_name = z.name

    batch_size, zr, zc, ch = z.shape

    r, c = pool_shape

    zpart = []

    mx = None

    for i in xrange(r):
        zpart.append([])
        for j in xrange(c):
            cur_part = z[:,i:zr:r,j:zc:c,:]
            cur_part.name = z_name + '[%d,%d]' % (i,j)
            zpart[i].append( cur_part )
            if mx is None:
                mx = T.maximum(0.,cur_part)
                mx.name = 'max(0,'+cur_part.name+')'
            else:
                mx_name = 'max('+cur_part.name+','+mx.name+')'
                mx = T.maximum(mx,cur_part)
                mx.name = mx_name
    mx.name = 'local_max('+z_name+')'

    pt = []

    for i in xrange(r):
        pt.append([])
        for j in xrange(c):
            z_ij = zpart[i][j]
            safe = z_ij - mx
            safe.name = 'safe_z(%s)' % z_ij.name
            cur_pt = T.exp(safe)
            cur_pt.name = 'pt(%s)' % z_ij.name
            pt[-1].append( cur_pt )

    off_pt = T.exp(0.-mx)
    off_pt.name = 'p_tilde_off(%s)' % z_name
    denom = off_pt

    for i in xrange(r):
        for j in xrange(c):
            denom = denom + pt[i][j]
    denom.name = 'denom(%s)' % z_name

    p = 1. - off_pt / denom
    p.name = 'p(%s)' % z_name

    hpart = []
    for i in xrange(r):
        hpart.append( [ pt_ij / denom for pt_ij in pt[i] ] )

    h = T.alloc(0., batch_size, zr, zc, ch)

    for i in xrange(r):
        for j in xrange(c):
            h = T.set_subtensor(h[:,i:zr:r,j:zc:c,:],hpart[i][j])

    h.name = 'h(%s)' % z_name

    return p, h

def check_correctness(f):
    rng = np.random.RandomState([2012,7,19])
    batch_size = 5
    rows = 32
    cols = 30
    channels = 3
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn( batch_size, rows, cols, channels ).astype(config.floatX)

    p_np, h_np = max_pool_python( zv, (pool_rows, pool_cols) )

    z_th = T.TensorType( broadcastable=(False,False,False,False), dtype = config.floatX)()
    z_th.name = 'z_th'

    p_th, h_th = f( z_th, (pool_rows, pool_cols) )

    func = function([z_th],[p_th,h_th])

    pv, hv = func(zv)

    assert p_np.shape == pv.shape
    assert h_np.shape == hv.shape
    if not np.allclose(h_np,hv):
        print (h_np.min(),h_np.max())
        print (hv.min(),hv.max())
        assert False
    assert np.allclose(p_np,pv)
    print 'Correct'

def profile(f):
    rng = np.random.RandomState([2012,7,19])
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn( batch_size, rows, cols, channels ).astype(config.floatX)

    #put the inputs + outputs in shared variables so we don't pay GPU transfer during test
    p_shared = sharedX(zv[:,0:rows:pool_rows,0:cols:pool_cols,:])
    h_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f( z_shared, (pool_rows, pool_cols) )

    func = function([],updates = { p_shared : p_th, h_shared : h_th} )

    print 'warming up'
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        print t2 - t1
        results.append(t2-t1)
    print 'final: ',sum(results)/float(trials)

def profile_grad(f):
    rng = np.random.RandomState([2012,7,19])
    batch_size = 80
    rows = 26
    cols = 27
    channels = 30
    pool_rows = 2
    pool_cols = 3
    zv = rng.randn( batch_size, rows, cols, channels ).astype(config.floatX)

    #put the inputs + outputs in shared variables so we don't pay GPU transfer during test
    grad_shared = sharedX(zv)
    z_shared = sharedX(zv)

    p_th, h_th = f( z_shared, (pool_rows, pool_cols) )

    func = function([],updates = { grad_shared : T.grad(p_th.sum() +  h_th.sum(), z_shared)} )

    print 'warming up'
    for i in xrange(10):
        func()

    trials = 10
    results = []

    for i in xrange(trials):
        t1 = time.time()
        for j in xrange(10):
            func()
        t2 = time.time()
        print t2 - t1
        results.append(t2-t1)
    print 'final: ',sum(results)/float(trials)

if __name__ == '__main__':
    check_correctness(max_pool)
    profile(max_pool)
    profile(max_pool_raw)
    profile_grad(max_pool)
    profile_grad(max_pool_raw)






