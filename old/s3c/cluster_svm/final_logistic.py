#TODO: support concatenating multiple datasets
print 'script launched'

try:
    from ia3n.util.mem import MemoryMonitor
    mem = MemoryMonitor()
except ImportError:
    mem = None
if mem:
    print 'memory usage on launch: '+str(mem.usage())
import numpy as np
from optparse import OptionParser
from galatea.s3c.hacky_multiclass_logistic import HackyMulticlassLogistic
from galatea.s3c.feature_loading import get_features
from pylearn2.utils import serial
from pylearn2.datasets.cifar10 import CIFAR10
from pylearn2.datasets.cifar100 import CIFAR100
import gc
gc.collect()
if mem:
    print 'memory usage after imports'+str(mem.usage())

def train_model(fold_train_X, fold_train_y, C):
    fold_train_X = np.cast['float64'](fold_train_X)

    model = HackyMulticlassLogistic(C).fit(fold_train_X, fold_train_y)
    gc.collect()

    return model

def get_labels_and_fold_indices(cifar10, cifar100, stl10):
    assert stl10 or cifar10 or cifar100
    assert stl10+cifar10+cifar100 == 1

    if stl10:
        print 'loading entire stl-10 train set just to get the labels and folds'
        stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/train.pkl")
        train_y = stl10.y

        fold_indices = stl10.fold_indices
    elif cifar10 or cifar100:
        if cifar10:
            print 'loading entire cifar10 train set just to get the labels'
            cifar = CIFAR10(which_set = 'train')
        else:
            assert cifar100
            print 'loading entire cifar100 train set just to get the labels'
            cifar = CIFAR100(which_set = 'train')
            cifar.y = cifar.y_fine
        train_y = cifar.y
        assert train_y is not None

        fold_indices = np.zeros((5,40000),dtype='uint16')
        idx_list = np.cast['uint16'](np.arange(1,50001)) #mimic matlab format of stl10
        for i in xrange(5):
            mask = idx_list < i * 10000 + 1
            mask += idx_list >= (i+1) * 10000 + 1
            fold_indices[i,:] = idx_list[mask]
        assert fold_indices.min() == 1
        assert fold_indices.max() == 50000


    return train_y, fold_indices


def main(train_path,
        out_path,
        dataset,
        standardize,
        C,
        **kwargs):



    stl10 = dataset == 'stl10'
    cifar10 = dataset == 'cifar10'
    cifar100 = dataset == 'cifar100'
    assert stl10 + cifar10 + cifar100 == 1

    print 'getting labels and oflds'
    if mem:
        print 'mem usage before getting labels and folds '+str(mem.usage())
    train_y, fold_indices = get_labels_and_fold_indices(cifar10, cifar100, stl10)
    if mem:
        print 'mem usage after getting labels and folds '+str(mem.usage())
    gc.collect()
    assert train_y is not None

    print 'loading training features'
    train_X = get_features(train_path, split = False, standardize = standardize)


    assert str(train_X.dtype) == 'float32'
    if stl10:
        assert train_X.shape[0] == 5000
    if cifar10 or cifar100:
        assert train_X.shape[0] == 50000
        assert train_y.shape == (50000,)

    model = train_model(train_X, train_y, C, **kwargs)

    serial.save(out_path, model)

if __name__ == '__main__':
    print '__main__ detected'

    parser = OptionParser()
    parser.add_option("-d", "--train",
                action="store", type="string", dest="train")
    parser.add_option("-o", "--out",
                action="store", type="string", dest="out")
    parser.add_option('-C', type='float', dest='C', action='store', default = None)
    parser.add_option('--dataset', type='string', dest = 'dataset', action='store', default = None)
    parser.add_option('--standardize',action="store_true", dest="standardize", default=False)

    (options, args) = parser.parse_args()

    assert options.dataset is not None
    assert options.C is not None
    assert options.out is not None


    print 'parsed the args'
    main(train_path=options.train,
         out_path = options.out,
         C = options.C,
         dataset = options.dataset,
         standardize = options.standardize,
    )

