#TODO: support concatenating multiple datasets
try:
    from ia3n.util.mem import MemoryMonitor
    mem = MemoryMonitor()
except ImportError:
    mem = None
import numpy as np
import warnings
from optparse import OptionParser
try:
    from sklearn.svm import LinearSVC, SVC
except ImportError:
    from scikits.learn.svm import LinearSVC, SVC
from galatea.s3c.feature_loading import get_features
from pylearn2.utils import serial
from pylearn2.datasets.cifar10 import CIFAR10
import gc


rng = np.random.RandomState([1,2,3])


def get_svm_type(C):
    svm_type = LinearSVC(C=C)
    return svm_type


def subtrain(fold_train_X, fold_train_y, C):
    assert str(fold_train_X.dtype) == 'float32'

    #assert fold_train_X.flags.c_contiguous

    svm = get_svm_type(C).fit(fold_train_X, fold_train_y)
    gc.collect()
    return svm

def validate(train_X, train_y, fold_indices, C):
    train_mask = np.zeros((train_X.shape[0],),dtype='uint8')
    #Yes, Adam's site really does say to use the 1000 indices as the train,
    #not the validation set
    #The -1 is to convert from matlab indices
    train_mask[fold_indices-1] = 1

    svm = subtrain( train_X[train_mask.astype(bool),:], train_y[train_mask.astype(bool)], \
            C = C)
    gc.collect()


    this_fold_valid_X = train_X[(1-train_mask).astype(bool),:]
    y_pred = svm.predict(this_fold_valid_X)
    this_fold_valid_y = train_y[(1-train_mask).astype(bool)]

    rval = (this_fold_valid_y == y_pred).mean()

    return rval

def train(fold_indices, train_X, train_y, C_list,  skip_cv = False ):
    """
    :param type: one of 'linear' or 'rbf'
    """

    n_train = train_X.shape[0]

    # Train a SVM classification model

    if skip_cv:
        best_C ,= C_list
    else:
        print " Finding the best C"

        param_grid = { 'C': C_list, 'gamma': [0.] }

        best_acc = -1
        for C in param_grid['C']:
            print '  C=%f' % (C,)

            acc = 0.0
            for i in xrange(fold_indices.shape[0]):
                print '   fold ',i

                this_fold_acc = validate(train_X, train_y, fold_indices[i,:], C)
                gc.collect()

                print '    fold accuracy: %f' % (this_fold_acc,)
                acc += this_fold_acc / float(fold_indices.shape[0])
            print '   accuracy this C: %f' % (acc,)


            if acc > best_acc:
                print '    best so far!'
                best_acc = acc
                best_C = C

        print " Validation set accuracy: ",best_acc

    print " Training final svm"
    final_svm = get_svm_type(best_C).fit(train_X, train_y)

    return final_svm

def get_labels_and_fold_indices(cifar10, stl10):
    assert stl10 or cifar10
    assert not (stl10 and cifar10)
    if stl10:
        print 'loading entire stl-10 train set just to get the labels and folds'
        stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/train.pkl")
        train_y = stl10.y

        fold_indices = stl10.fold_indices
    elif cifar10:
        print 'loading entire cifar10 train set just to get the labels'
        cifar10 = CIFAR10(which_set = 'train')
        train_y = cifar10.y
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


def get_test_labels(cifar10, stl10):
    assert cifar10 or stl10
    assert not (cifar10 and stl10)

    if stl10:
        print 'loading entire stl-10 test set just to get the labels'
        stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/test.pkl")
        return stl10.y
    if cifar10:
        print 'loading entire cifar10 test set just to get the labels'
        cifar10 = CIFAR10(which_set = 'test')
        return np.asarray(cifar10.y)

def random_subset(train_X, train_y, num_examples):
    m = 50000
    assert train_X.shape[0] == m
    assert train_y.shape == (m,)

    min_class = train_y.min()
    max_class = train_y.max()

    classes = range(min_class,max_class+1)

    class_idxs = []

    for i in xrange(len(classes)):
        c = classes[i]
        this_class_idxs = np.nonzero( train_y == c)
        assert isinstance(this_class_idxs,tuple)
        assert len(this_class_idxs) == 1
        this_class_idxs ,= this_class_idxs
        assert len(this_class_idxs.shape) == 1
        class_idxs.append(this_class_idxs)

    assert len(classes) == 10

    taken = np.zeros( (m, ), dtype='uint8')

    out_m = 10 * num_examples

    idxs = np.zeros((out_m,),dtype='uint16')

    idx = 0
    for i in xrange(num_examples):
        for j in xrange(len(classes)):
            found = False
            while not found:
                k = rng.randint(0,len(class_idxs[j]))
                k = class_idxs[j][k]
                found = not taken[k]
            taken[k] = 1
            idxs[idx] = k

            idx += 1

    assert idx == out_m

    train_X = train_X[idxs,:]
    train_y = train_y[idxs]

    return train_X, train_y

def run_experiment(train_X, train_y, test_X, test_y, num_examples, C_list):
    print 'running a new experiment'
    assert num_examples % 5 == 0
    fold_size = (4 * num_examples ) / 5
    test_fold_size = num_examples / 5

    fold_indices = np.zeros((5,fold_size),dtype='uint16')
    idx_list = np.cast['uint16'](np.arange(1,num_examples+1))
    for i in xrange(5):
        mask = idx_list < i * test_fold_size + 1
        mask += idx_list >= (i+1) * test_fold_size + 1
        fold_indices[i,:] = idx_list[mask]
    assert fold_indices.min() == 1
    assert fold_indices.max() == num_examples


    train_X, train_y = random_subset(train_X, train_y, num_examples)

    model = train(fold_indices, train_X, train_y, C_list)

    y_pred = model.predict(test_X)

    acc = (test_y == y_pred).mean()
    print ' test acc ',acc

    return acc



def main(train_path,
        test_path,
        num_examples,
        **kwargs):

    train_y, fold_indices = get_labels_and_fold_indices(cifar10 = True, stl10 = False)
    del fold_indices
    assert train_y is not None

    train_X = get_features(train_path, split = False)

    assert str(train_X.dtype) == 'float32'
    assert train_X.shape[0] == 50000
    assert train_y.shape == (50000,)

    test_X = get_features(test_path, split = False)
    test_y  = get_test_labels(cifar10 = True, stl10 = False)

    accs = []

    while True:
        accs.append(run_experiment(train_X,train_y,test_X, test_y,num_examples,**kwargs))
        v = np.asarray(accs)
        mn = v.mean()
        sd = v.std()
        print 'accuracy: %f +- %f' % (mn, sd)

if __name__ == '__main__':
    """
    Useful for quick tests.
    Usage: python train_bilinear.py
    """

    parser = OptionParser()
    parser.add_option('-n', "--num-train", action="store", type="int", dest="num_examples", default=-1)
    parser.add_option("-d", "--train",
                action="store", type="string", dest="train")
    parser.add_option("-t", "--test",
                action="store", type="string", dest="test")
    parser.add_option('-C', type='string', dest='C_list', action='store', default= '.1,.2,.5,1.')

    (options, args) = parser.parse_args()

    C_list = [ float(chunk) for chunk in options.C_list.split(',') ]

    main(train_path=options.train,
         test_path = options.test,
         C_list = C_list,
         num_examples = options.num_examples
    )
