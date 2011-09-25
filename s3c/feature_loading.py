from pylearn2.utils import serial
import numpy as np
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter

def get_features(path, split):
    if path.endswith('.npy'):
        topo_view = np.load(path)
    else:
        topo_view = serial.load(path)

    view_converter = DefaultViewConverter(topo_view.shape[1:])

    print 'converting data'
    X = view_converter.topo_view_to_design_mat(topo_view)

    if split:
        X = np.concatenate( (np.abs(X),np.abs(-X)), axis=1)

    return X

