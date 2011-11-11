#TODO: support concatenating multiple datasets
try:
    from ia3n.util.mem import MemoryMonitor
    mem = MemoryMonitor()
except ImportError:
    mem = None
if mem:
    print 'memory usage on launch: '+str(mem.usage())
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
from pylearn2.datasets.cifar100 import CIFAR100
import gc
gc.collect()
if mem:
    print 'memory usage after imports'+str(mem.usage())

def get_svm_type(C, one_against_many):
    if one_against_many:
        svm_type = LinearSVC(C=C)
    else:
        svm_type =  SVC(kernel='linear',C=C)
    return svm_type


def subtrain(fold_train_X, fold_train_y, C, one_against_many):
    assert str(fold_train_X.dtype) == 'float32'

    #assert fold_train_X.flags.c_contiguous

    if mem:
        print 'mem usage before calling fit: '+str(mem.usage())
    svm = get_svm_type(C, one_against_many).fit(fold_train_X, fold_train_y)
    gc.collect()
    if mem:
        print 'mem usage after calling fit: '+str(mem.usage())
    return svm

def validate(train_X, train_y, fold_indices, C, one_against_many):
    train_mask = np.zeros((train_X.shape[0],),dtype='uint8')
    #Yes, Adam's site really does say to use the 1000 indices as the train,
    #not the validation set
    #The -1 is to convert from matlab indices
    train_mask[fold_indices-1] = 1

    if mem:
        print 'mem usage before calling subtrain: '+str(mem.usage())
    svm = subtrain( train_X[train_mask.astype(bool),:], train_y[train_mask.astype(bool)], \
            C = C, one_against_many = one_against_many)
    gc.collect()
    if mem:
        print 'mem usage after calling subtrain: '+str(mem.usage())


    this_fold_valid_X = train_X[(1-train_mask).astype(bool),:]
    y_pred = svm.predict(this_fold_valid_X)
    this_fold_valid_y = train_y[(1-train_mask).astype(bool)]

    rval = (this_fold_valid_y == y_pred).mean()

    return rval

def train(fold_indices, train_X, train_y, report, C_list, one_against_many=False, skip_cv = False , max_folds = None):
    """
    :param type: one of 'linear' or 'rbf'
    """

    n_train = train_X.shape[0]

    # Train a SVM classification model

    if skip_cv:
        best_C ,= C_list
        report.add_validation_result('C='+str(best_C),-1.)
    else:
        print "Finding the best C"

        param_grid = { 'C': C_list, 'gamma': [0.] }

        num_folds = fold_indices.shape[0]
        if max_folds is not None and num_folds > max_folds:
            num_folds = max_folds

        best_acc = -1
        for C in param_grid['C']:
            print ' C=%f' % (C,)

            acc = 0.0
            for i in xrange(num_folds):
                print '  fold ',i

                if mem:
                    print 'mem usage before calling validate:'+str(mem.usage())
                this_fold_acc = validate(train_X, train_y, fold_indices[i,:], C, one_against_many)
                gc.collect()
                if mem:
                    print 'mem usage after calling validate:'+str(mem.usage())

                print '   fold accuracy: %f' % (this_fold_acc,)
                acc += this_fold_acc / float(num_folds)
            print '  accuracy this C: %f' % (acc,)

            report.add_validation_result('C='+str(C),acc)

            if acc > best_acc:
                print '   best so far!'
                best_acc = acc
                best_C = C

        print "Validation set accuracy: ",best_acc

    print "Training final svm"
    final_svm = get_svm_type(best_C, one_against_many).fit(train_X, train_y)

    return final_svm

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


class Report:
    def __init__(self, train_path, split, stl10, cifar10, cifar100):
        self.train_path = train_path
        self.split = split
        assert stl10 or cifar10 or cifar100
        assert stl10 + cifar10 + cifar100 == 1
        self.stl10 = stl10
        self.cifar10 = cifar10
        self.cifar100 = cifar100
        self.desc_to_acc = {}

    def add_validation_result(self, hparam_desc, acc):
        self.desc_to_acc[hparam_desc] = acc

    def write(self, out_path):

        f = open(out_path,'w')

        if self.stl10:
            f.write('STL-10 SVM Cross Validation report\n')
        elif self.cifar10:
            f.write('CIFAR10 SVM Cross Validation report\n')
        elif self.cifar100:
            f.write('CIFAR100-fine SVM Cross Validation report\n')
        f.write('Training features: '+self.train_path+'\n')
        f.write('Splitting enabled: '+str(self.split)+'\n')

        best_acc = max(self.desc_to_acc.values())

        f.write('Validation set accuracy: '+str(best_acc)+'\n')

        for desc in self.desc_to_acc:
            f.write(desc+': '+str(self.desc_to_acc[desc])+'\n')

        f.close()

def main(train_path,
        out_path,
        split,
        dataset,
        standardize,
        **kwargs):

    stl10 = dataset == 'stl10'
    cifar10 = dataset == 'cifar10'
    cifar100 = dataset == 'cifar100'
    assert stl10 + cifar10 + cifar100 == 1

    if mem:
        print 'mem usage before getting labels and folds '+str(mem.usage())
    train_y, fold_indices = get_labels_and_fold_indices(cifar10, cifar100, stl10)
    if mem:
        print 'mem usage after getting labels and folds '+str(mem.usage())
    gc.collect()
    assert train_y is not None

    print 'loading training features'

    if mem:
        print 'mem usage before getting features '+str(mem.usage())
    train_X = get_features(train_path, split, standardize)
    #assert train_X.flags.c_contiguous
    gc.collect()
    if mem:
        print 'mem usage after getting features '+str(mem.usage())


    assert str(train_X.dtype) == 'float32'
    if stl10:
        assert train_X.shape[0] == 5000
    if cifar10 or cifar100:
        assert train_X.shape[0] == 50000
        assert train_y.shape == (50000,)

    report = Report(train_path, split, stl10, cifar10, cifar100)

    gc.collect()

    if mem:
        print 'mem usage before calling train: '+str(mem.usage())
    model = train(fold_indices, train_X, train_y, report, **kwargs)


    serial.save(out_path+'.model.pkl', model)
    report.write(out_path+'.validation_report.txt')

if __name__ == '__main__':
    """
    Useful for quick tests.
    Usage: python train_bilinear.py
    """

    parser = OptionParser()
    parser.add_option("-d", "--train",
                action="store", type="string", dest="train")
    parser.add_option("-o", "--out",
                action="store", type="string", dest="out")
    parser.add_option("--one-against-one", action="store_false", dest="one_against_many", default=True,
                      help="use a one-against-one classifier rather than a one-against-many classifier")
    parser.add_option("--split", action="store_true", dest="split", default = False, help="double the example size by splitting each feature into a positive component and a negative component")
    parser.add_option('-C', type='string', dest='C_list', action='store', default= '.01,.02,.05,.1,.15,.2,.5,1,5,10')
    parser.add_option('--dataset', type='string', dest = 'dataset', action='store', default = None)
    parser.add_option('--skip-cv', action="store_true", dest="skip_cv", default=False, help="don't cross validate, just use first value in C list")
    parser.add_option('--max-folds',action="store_true", dest="max_folds", default=None)
    parser.add_option('--standardize',action="store_true", dest="standardize", default=False)

    (options, args) = parser.parse_args()

    assert options.dataset
    C_list = [ float(chunk) for chunk in options.C_list.split(',') ]

    main(train_path=options.train,
         out_path = options.out,
         one_against_many = options.one_against_many,
         split = options.split,
         C_list = C_list,
         dataset = options.dataset,
         skip_cv = options.skip_cv,
         max_folds = options.max_folds,
         standardize = options.standardize
    )
