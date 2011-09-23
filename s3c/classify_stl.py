#TODO: support concatenating multiple datasets
#TODO: support recursive refinement of C search
#TODO: redo scopes so fewer dels are needed

# allows to plot figures without X11 (i.e on cluster)
import time
import numpy as np
import warnings
import os
import matplotlib
import pylearn.datasets.MNIST
matplotlib.use("Agg")

from optparse import OptionParser

from theano import function
import theano.tensor as T
from jobman import make
import pylearn.datasets.icml07
from scikits.learn.grid_search import GridSearchCV
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from scikits.learn.metrics import classification_report
from scikits.learn.metrics import confusion_matrix
from scikits.learn.svm import LinearSVC, SVC

from pylearn2.utils import serial

def get_svm_type(C, one_against_many):
    if one_against_many:
        svm_type = LinearSVC(C=C)
    else:
        svm_type =  SVC(kernel='linear',C=C)
    return svm_type

def validate(train_X, train_y, fold_indices, C, one_against_many):
    train_mask = np.zeros((5000,),dtype='uint8')
    #Yes, Adam's site really does say to use the 1000 indices as the train,
    #not the validation set
    #The -1 is to convert from matlab indices
    train_mask[fold_indices-1] = 1

    this_fold_train_X = train_X[train_mask.astype(bool),:]
    this_fold_train_y = train_y[train_mask.astype(bool)]
    svm = get_svm_type(C, one_against_many).fit(this_fold_train_X,this_fold_train_y)
    del this_fold_train_X
    del this_fold_train_y

    this_fold_valid_X = train_X[(1-train_mask).astype(bool),:]
    y_pred = svm.predict(this_fold_valid_X)
    del this_fold_valid_X
    this_fold_valid_y = train_y[(1-train_mask).astype(bool)]

    rval = (this_fold_valid_y == y_pred).mean()

    return rval

def train(fold_indices, train_X, train_y,  one_against_many=False ):
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

        if acc > best_acc:
            print '   best so far!'
            best_acc = acc
            best_C = C

    print "Validation set accuracy: ",best_acc

    print "Training final svm"
    final_svm = get_svm_type(best_C, one_against_many).fit(train_X, train_y)

    return final_svm

def test(model, X, y):
    print "Evaluating svm"
    y_pred = model.predict(X)
    try:
        print "Accuracy ",(y == y_pred).mean()
        print classification_report(y, y_pred)#, labels=selected_target,
                                #class_names=category_names[selected_target])

        print confusion_matrix(y, y_pred)#, labels=selected_target)
    except:
        print "something went wrong"
        print 'y:'
        print y
        print 'y_pred:'
        print y_pred
        print 'extra info'
        print type(y)
        print type(y_pred)
        print y.dtype
        print y_pred.dtype
        print y.shape
        print y_pred.shape
#


def get_labels_and_fold_indices():
    print 'loading entire stl-10 train set just to get the labels and folds'
    stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/train.pkl")
    train_y = stl10.y

    fold_indices = stl10.fold_indices

    return train_y, fold_indices

def get_test_labels():
    print 'loading entire stl-10 test set just to get the labels'
    stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/test.pkl")
    return stl10.y

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

def main(train_path,
        test_path,
        split,
        **kwargs):

    train_y, fold_indices = get_labels_and_fold_indices()

    print 'loading training features'
    train_X = get_features(train_path, split)
    assert train_X.shape[0] == 5000

    model = train(fold_indices, train_X, train_y, **kwargs)

    del train_X
    del train_y

    y = get_test_labels()
    X = get_features(test_path, split)
    assert X.shape[0] == 8000

    test(model,X,y)


if __name__ == '__main__':
    """
    Useful for quick tests.
    Usage: python train_bilinear.py
    """

    parser = OptionParser()
    parser.add_option("-d", "--train",
                action="store", type="string", dest="train")
    parser.add_option("-t", "--test",
                action="store", type="string", dest="test")
    parser.add_option("--one-against-one", action="store_false", dest="one_against_many", default=True,
                      help="use a one-against-one classifier rather than a one-against-many classifier")
    parser.add_option("--split", action="store_true", dest="split", default = False, help="double the example size by splitting each feature into a positive component and a negative component")


    (options, args) = parser.parse_args()

    main(train_path=options.train,
         test_path=options.test,
         one_against_many = options.one_against_many,
         split = options.split
    )
