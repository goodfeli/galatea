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
from pylearn2.datasets.tl_challenge import TL_Challenge
from galatea.s3c.feature_loading import get_features
from pylearn2.datasets.cifar100 import CIFAR100
from pylearn2.utils import serial
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

def validate(train_X, train_y, test_X, test_y, C, one_against_many):
    svm = subtrain( train_X, train_y, \
            C = C, one_against_many = one_against_many)

    y_pred = svm.predict(test_X)

    rval = (test_y == y_pred).mean()

    return rval

def train(train_X, train_y, test_X, test_y, report, C_list, one_against_many=False, skip_cv = False ):
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

        best_acc = -1
        for C in param_grid['C']:
            print ' C=%f' % (C,)

            acc = 0.0
            for i in xrange(1):
                print '  fold ',i

                if mem:
                    print 'mem usage before calling validate:'+str(mem.usage())
                this_fold_acc = validate(train_X, train_y, test_X, test_y, C, one_against_many)
                gc.collect()
                if mem:
                    print 'mem usage after calling validate:'+str(mem.usage())

                print '   fold accuracy: %f' % (this_fold_acc,)
                acc += this_fold_acc
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

def get_labels():

    cifar100 = CIFAR100(which_set = "train")
    train_y = cifar100.y_coarse

    assert train_y.shape == (50000,)

    for i in xrange(50000):
        if train_y[i] not in [3, 4, 6, 7, 11, 12]:
            train_y[i] = 0

    tlc = TL_Challenge(which_set = 'train')
    test_y = tlc.y_coarse

    return train_y, test_y


class Report:
    def __init__(self, train_path, split):
        self.train_path = train_path
        self.split = split
        self.desc_to_acc = {}

    def add_validation_result(self, hparam_desc, acc):
        self.desc_to_acc[hparam_desc] = acc

    def write(self, out_path):

        f = open(out_path,'w')

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
        **kwargs):


    if mem:
        print 'mem usage before getting labels and folds '+str(mem.usage())
    train_y, test_y = get_labels()
    if mem:
        print 'mem usage after getting labels and folds '+str(mem.usage())
    gc.collect()
    assert train_y is not None

    print 'loading training features'

    if mem:
        print 'mem usage before getting features '+str(mem.usage())
    train_X = get_features(train_path.replace('aux','train'),split)
    test_X = get_features(train_path, split)
    #assert train_X.flags.c_contiguous
    gc.collect()
    if mem:
        print 'mem usage after getting features '+str(mem.usage())

    assert train_X.shape[0] == 50000
    assert train_y.shape == (50000,)

    assert str(train_X.dtype) == 'float32'
    assert test_X.shape[0] == 120
    assert test_y.shape == (120,)

    report = Report(train_path, split)

    gc.collect()

    if mem:
        print 'mem usage before calling train: '+str(mem.usage())
    model = train(train_X, train_y, test_X, test_y, report, **kwargs)


    serial.save(out_path+'.model.pkl', model)
    report.write(out_path+'.validation_report.txt')

if __name__ == '__main__':
    """
    Useful for quick tests.
    Usage: python train_bilinear.py
    """

    parser = OptionParser()
    parser.add_option("--one-against-one", action="store_false", dest="one_against_many", default=True,
                      help="use a one-against-one classifier rather than a one-against-many classifier")
    parser.add_option("--split", action="store_true", dest="split", default = False, help="double the example size by splitting each feature into a positive component and a negative component")
    parser.add_option('-C', type='string', dest='C_list', action='store', default= '10,25,50,100,1000')
    parser.add_option('--skip-cv', action="store_true", dest="skip_cv", default=False, help="don't cross validate, just use first value in C list")

    (options, args) = parser.parse_args()

    C_list = [ float(chunk) for chunk in options.C_list.split(',') ]

    main(train_path='/u/goodfeli/galatea/s3c/config/TLC/extract/A_exp_h_3_aux.npy',
         out_path = 'course_analysis_',
         one_against_many = options.one_against_many,
         split = options.split,
         C_list = C_list,
         skip_cv = options.skip_cv
    )
