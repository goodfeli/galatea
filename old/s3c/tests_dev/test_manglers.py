from galatea.s3c.s3c import numpy_norm_clip
from galatea.s3c.s3c import theano_norm_clip
from galatea.s3c.s3c import numpy_norms
import numpy as np
from theano import function
from theano import shared

def test_norm_clip():

    rng = np.random.RandomState([1,2,3])

    D = 3
    H = 50

    W = rng.rand(D,H)

    norms = numpy_norms(W)

    min_idx = rng.randint(0,H/5)
    max_idx = rng.randint(4*H/5,H+1)

    sorted_norms = sorted(norms)

    min_norm = sorted_norms[min_idx]
    max_norm = sorted_norms[max_idx]

    nW = numpy_norm_clip(W,min_norm,max_norm)
    tW = function([],theano_norm_clip(shared(W),min_norm,max_norm))()

    assert np.allclose(nW,tW)

    unchanged_mask = (norms >= min_norm) * (norms <= max_norm)

    assert np.allclose(nW[:, unchanged_mask],W[:,unchanged_mask])

    new_norms = numpy_norms(nW)

    assert np.abs(  new_norms.min() - min_norm) < 1e-5
    assert np.abs(  new_norms.max() - max_norm) < 1e-5
