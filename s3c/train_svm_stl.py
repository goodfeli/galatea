#TODO: support concatenating multiple datasets

import numpy as np
import warnings
from optparse import OptionParser
from scikits.learn.svm import LinearSVC, SVC
from galatea.s3c.feature_loading import get_features
from pylearn2.utils import serial

def get_svm_type(C, one_against_many):
    if one_against_many:
        svm_type = LinearSVC(C=C)
    else:
        svm_type =  SVC(kernel='linear',C=C)
    return svm_type


def subtrain(fold_train_X, fold_train_y, C, one_against_many):
    svm = get_svm_type(C, one_against_many).fit(fold_train_X, fold_train_y)
    return svm

def validate(train_X, train_y, fold_indices, C, one_against_many):
    train_mask = np.zeros((5000,),dtype='uint8')
    #Yes, Adam's site really does say to use the 1000 indices as the train,
    #not the validation set
    #The -1 is to convert from matlab indices
    train_mask[fold_indices-1] = 1

    svm = subtrain( train_X[train_mask.astype(bool),:], train_y[train_mask.astype(bool)], \
            C = C, one_against_many = one_against_many)

    this_fold_valid_X = train_X[(1-train_mask).astype(bool),:]
    y_pred = svm.predict(this_fold_valid_X)
    this_fold_valid_y = train_y[(1-train_mask).astype(bool)]

    rval = (this_fold_valid_y == y_pred).mean()

    return rval

def train(fold_indices, train_X, train_y, report, one_against_many=False ):
    """
    :param type: one of 'linear' or 'rbf'
    """

    n_train = train_X.shape[0]

    # Train a SVM classification model
    print "Finding the best C"

    param_grid = { 'C': [.01, .02, .1, .2, 1, 5, 10, 50, 100], 'gamma': [0.] }


    warnings.warn("""
    scikits GridSearchCV seemed to fail for bilinear rbm project so here we just
    use our own code (should we really trust scikits svm when they can't even implement grid search right? )
    """)

    best_acc = -1
    for C in param_grid['C']:
        print ' C=%f' % (C,)

        acc = 0.0
        for i in xrange(fold_indices.shape[0]):
            print '  fold ',i

            this_fold_acc = validate(train_X, train_y, fold_indices[i,:], C, one_against_many)

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
    print 'loading entire stl-10 train set just to get the labels and folds'
    stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/train.pkl")
    train_y = stl10.y

    fold_indices = stl10.fold_indices

    return train_y, fold_indices


class Report:
    def __init__(self, train_path, split):
        self.train_path = train_path
        self.split = split
        self.desc_to_acc = {}

    def add_validation_result(self, hparam_desc, acc):
        self.desc_to_acc[hparam_desc] = acc

    def write(self, out_path):

        f = open(out_path,'w')

        f.write('STL-10 SVM Cross Validation report'+'\n')
        f.write('Training features: '+self.train_path+'\n')
        f.write('Splitting enabled: '+str(self.split)+'\n')

        best_acc = max(self.desc_to_acc)

        f.write('Validation set accuracy: '+str(best_acc)+'\n')

        for desc in self.desc_to_acc:
            f.write(desc+': '+str(self.desc_to_acc[desc])+'\n')

        f.close()

def main(train_path,
        out_path,
        split,
        **kwargs):

    train_y, fold_indices = get_labels_and_fold_indices()

    print 'loading training features'
    train_X = get_features(train_path, split)
    assert train_X.shape[0] == 5000

    report = Report(train_path, split)

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


    (options, args) = parser.parse_args()

    main(train_path=options.train,
         out_path = options.out,
         one_against_many = options.one_against_many,
         split = options.split
    )
