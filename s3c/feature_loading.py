from pylearn2.utils import serial
import numpy as np
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter

def get_features(path, split, standardize):
    if path.find(',') != -1:
        paths = path.split(',')
        Xs = [ get_features(subpath, split, standardize) for subpath in paths ]
        X = np.concatenate( Xs, axis = 1)
        return X


    if path.endswith('.npy'):
        topo_view = np.load(path)
    else:
        topo_view = serial.load(path)

        if str(type(topo_view)).find('h5py') != -1:
            name ,= topo_view.keys()
            topo_view = topo_view[name].value.T

    if len(topo_view.shape) == 2:
        X = topo_view
    else:
        view_converter = DefaultViewConverter(topo_view.shape[1:])

        print 'converting data'
        X = view_converter.topo_view_to_design_mat(topo_view)

    if split:
        X = np.concatenate( (np.abs(X),np.abs(-X)), axis=1)

    if standardize:
        assert False   #bug: if X is test set, we need to subtract train mean, divide by train std
        X -= X.mean(axis=0)
        X /= np.sqrt(.01+np.var(X,axis=0))

    return X

