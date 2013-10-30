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
from pylearn2.utils import serial
from pylearn2.datasets.cifar10 import CIFAR10
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
    return svm

def validate(train_X, y_fine, y_coarse, fold_indices, omnivore_classifier,
        fruit_classifier, C, one_against_many):

    train_mask = np.zeros((train_X.shape[0],),dtype='uint8')
    train_mask[fold_indices-1] = 1

    svm = subtrain( train_X[train_mask.astype(bool),:], y_coarse[train_mask.astype(bool)], \
            C = C, one_against_many = one_against_many)

    this_fold_valid_X = train_X[(1-train_mask).astype(bool),:]
    y_pred = svm.predict(this_fold_valid_X)
    this_fold_valid_y = y_fine[(1-train_mask).astype(bool)]


    assert y_pred.shape == (1,)
    assert this_fold_valid_y.shape == (1,)

    y_pred = y_pred[0]

    if y_pred == 4:
        y_pred = fruit_classifier.predict(this_fold_valid_X)[0]
        assert y_pred in [104,106,108]
    elif y_pred == 11:
        y_pred = omnivore_classifier.predict(this_fold_valid_X)[0]
        assert y_pred in [101,102,103]
    else:
        d = {   3  : 105,
                12 : 107,
                7  : 100,
                6  : 109 }
        y_pred = d[y_pred]


    this_fold_valid_y = this_fold_valid_y[0]

    rval = float(this_fold_valid_y == y_pred)

    return rval

def train(fold_indices, omnivore_classifiers, fruit_classifiers, train_X, y_fine, y_coarse, report, C_list, one_against_many=False, skip_cv = False ):
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
            for i in xrange(fold_indices.shape[0]):
                print '  fold ',i

                this_fold_acc = validate(train_X, y_fine, y_coarse, fold_indices[i,:],
                        omnivore_classifiers[i], fruit_classifiers[i], C, one_against_many)

                print '   fold accuracy: %f' % (this_fold_acc,)
                acc += this_fold_acc / float(fold_indices.shape[0])
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

def get_labels_and_fold_indices():

    tlc = TL_Challenge(which_set = 'train')
    train_y = tlc.y_fine
    assert train_y is not None

    fold_indices = np.zeros((120,119),dtype='uint16')
    idx_list = np.cast['uint16'](np.arange(1,121)) #mimic matlab format of stl10
    for i in xrange(120):
        mask = idx_list < i  + 1
        mask += idx_list >= (i+1)  + 1
        fold_indices[i,:] = idx_list[mask]
    assert fold_indices.min() == 1
    assert fold_indices.max() == 120


    return tlc.y_fine, tlc.y_coarse, fold_indices


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


def get_classifiers(name, train_X, y_fine, y_coarse, fold_indices):


    if name == 'omnivore':
        coarse_class = 11
        C = 100
    elif name == 'fruit':
        coarse_class = 4
        C = 100
    else:
        assert False

    classifiers = []

    for i in xrange(120):
        if y_coarse[i] == 4:
            assert y_fine[i] in [104,106,108]


    for i in xrange(fold_indices.shape[0]):
        #restrict train set to fold indices
        train_mask = np.zeros((train_X.shape[0],),dtype='uint8')
        train_mask[fold_indices[i,:]-1] = 1
        #fold_train_X = train_X[train_mask.astype(bool),:]
        #fold_y_fine = y_fine[train_mask.astype(bool)]
        #fold_y_coarse = y_coarse[train_mask.astype(bool)]

        #if coarse_class == 4:
            #for i in xrange(fold_y_fine.shape[0]):
                #if fold_y_coarse[i] == coarse_class:
                    #            assert fold_y_fine[i] in [104,106,108]

        #restrict train set to elements of coarse class
        fuckyou = (train_mask  * (y_coarse == coarse_class))
        fold_train_X = train_X[fuckyou.astype(bool),:]
        fold_train_y = y_fine[fuckyou.astype(bool)]

        m = fuckyou.sum()

        assert fold_train_y.shape == (m,)

        #fold_train_X = np.zeros((m,fold_train_X.shape[1]),dtype='float32')
        #fold_train_y = np.zeros((m,),dtype='uint8')

        if coarse_class == 4:
            for i in xrange(fold_train_y.shape[0]):
                assert fold_train_y[i] in [104,106,108]

        classifier = subtrain(fold_train_X, fold_train_y, C, False)

        classifiers.append(classifier)

    return classifiers


def main(train_path,
        out_path,
        split,
        **kwargs):

    y_fine, y_coarse, fold_indices = get_labels_and_fold_indices()

    gc.collect()

    print 'loading training features'

    train_X = get_features(train_path, split)
    #assert train_X.flags.c_contiguous
    gc.collect()


    assert str(train_X.dtype) == 'float32'
    assert train_X.shape[0] == 120
    assert y_fine.shape == (120,)
    assert y_coarse.shape == (120,)

    report = Report(train_path, split)

    gc.collect()


    print 'making omnivore classifiers'
    omnivore_classifiers = get_classifiers('omnivore',train_X,y_fine,y_coarse,fold_indices)
    print 'making fruit classifiers'
    fruit_classifiers = get_classifiers('fruit',train_X,y_fine,y_coarse,fold_indices)

    model = train(fold_indices, omnivore_classifiers, fruit_classifiers, train_X, y_fine, y_coarse, report, **kwargs)


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
    parser.add_option('-C', type='string', dest='C_list', action='store', default= '.5,1,5,10')
    parser.add_option('--skip-cv', action="store_true", dest="skip_cv", default=False, help="don't cross validate, just use first value in C list")

    (options, args) = parser.parse_args()

    C_list = [ float(chunk) for chunk in options.C_list.split(',') ]

    main(train_path='/u/goodfeli/galatea/s3c/config/TLC/extract/A_exp_h_3_aux.npy',
         out_path = 'tlc_cascade',
         one_against_many = options.one_against_many,
         split = options.split,
         C_list = C_list,
         skip_cv = options.skip_cv
    )
