import numpy as np

from pylearn2.datasets.binarizer import Binarizer as BinarizerDataset
from pylearn2.datasets.norb_small import FoveatedNORB

def russ_features(which_set, restrict_instances):

    raw = FoveatedNORB(which_set=which_set, restrict_instances=None, one_hot=True)

    X = np.load("/data/lisa/data/norb_small/ruslan_binarized/%s_X.npy" % which_set)
    assert raw.X.shape[0] == X.shape[0], (raw.X.shape[0], X.shape[0])
    raw.X = X

    y = np.load("/data/lisa/data/norb_small/ruslan_binarized/%s_Y.npy" % which_set)
    print y[0:5, :]
    print raw.y[0:5, :]
    assert np.allclose(y, raw.y)

    if restrict_instances is not None:
        raw.restrict_instances(restrict_instances)

    rval =  BinarizerDataset(raw)

    rval.X = raw.X

    return rval
