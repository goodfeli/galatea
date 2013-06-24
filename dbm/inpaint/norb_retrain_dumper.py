import gc
import numpy as np
import sys

from pylearn2.config import yaml_parse
from pylearn2.utils import serial

_, config_path = sys.argv
model = yaml_parse.load_path(config_path)

f = model.dump_func()

model.strip_down()
stripped_model_path = config_path.replace('.yaml', '_stripped.pkl')
serial.save(stripped_model_path, model)

srcs = {
        'train' : """!obj:pylearn2.datasets.norb_small.FoveatedNORB {
        which_set: "train",
        scale: 1,
        one_hot: 1
    }""",
        'test' : """!obj:pylearn2.datasets.norb_small.FoveatedNORB {
        which_set: "test",
        scale: 1,
        one_hot: 1
    }"""
        }

for which_set in srcs:
    gc.collect()
    dataset_src = srcs[which_set]

    dataset = yaml_parse.load(dataset_src)

    batch_size = 1000

    m = dataset.X.shape[0]

    X_pieces = [dataset.X[i:i+batch_size,:] for i in xrange(0, m, batch_size)]
    y_pieces = [dataset.y[i:i+batch_size,:] for i in xrange(0, m, batch_size)]
    del dataset
    gc.collect()
    X_pieces = map(f, X_pieces)
    gc.collect()

    X = np.concatenate(X_pieces, axis=0)
    y = np.concatenate(y_pieces, axis=0)

    del X_pieces
    del y_pieces

    out_path_X = config_path.replace('.yaml', '_' + which_set + '_X.npy')
    out_path_y = config_path.replace('.yaml', '_' + which_set + '_y.npy')

    np.save(out_path_X, X)
    np.save(out_path_y, y)










