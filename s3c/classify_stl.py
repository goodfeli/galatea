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

def classify(fold_indices, train_X, train_y, X,y,  one_against_many=False ):
    """
    :param type: one of 'linear' or 'rbf'
    """

    n_train = train_X.shape[0]

    # Train a SVM classification model
    print "Finding the best C"

    param_grid = { 'C': [.01, .02, .1, .2, 1, 5, 10, 50, 100], 'gamma': [0.] }

    def get_svm_type(  my_C):
        if one_against_many:
            svm_type = LinearSVC(C=my_C)
        else:
            svm_type =  SVC(kernel='linear',C=my_C,)
        return svm_type

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

            train_mask = np.zeros((5000,),dtype='uint8')
            #Yes, Adam's site really does say to use the 1000 indices as the train,
            #not the validation set
            train_mask[fold_indices[i,:]-1] = 1

            this_fold_train_X = train_X[train_mask.astype(bool),:]
            this_fold_train_y = train_y[train_mask.astype(bool)]
            svm = get_svm_type(C).fit(this_fold_train_X,this_fold_train_y)
            del this_fold_train_X
            del this_fold_train_y

            this_fold_valid_X = train_X[(1-train_mask).astype(bool),:]
            y_pred = svm.predict(this_fold_valid_X)
            del this_fold_valid_X
            this_fold_valid_y = train_y[(1-train_mask).astype(bool)]

            this_fold_acc = (y_pred == this_fold_valid_y).mean()
            del this_fold_valid_y
            print '   fold accuracy: %f' % (this_fold_acc,)
            acc += this_fold_acc / float(fold_indices.shape[0])
        print '  accuracy this C: %f' % (acc,)

        if acc > best_acc:
            print '   best so far!'
            best_acc = acc
            best_C = C

    print "Validation set accuracy: ",best_acc

    print "Training final svm"
    final_svm = get_svm_type(best_C).fit(train_X, train_y)

    print "Evaluating svm"
    y_pred = final_svm.predict(X)
    print classification_report(y, y_pred)#, labels=selected_target,
                                #class_names=category_names[selected_target])

    print confusion_matrix(y, y_pred)#, labels=selected_target)

    print "Accuracy ",(y == y_pred).mean()
#

def main(
        data_path,
        **kwargs):

    print 'loading entire stl-10 train set just to get the labels and folds'
    stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/train.pkl")
    train_y = stl10.y

    fold_indices = stl10.fold_indices
    del stl10

    print 'loading data'
    topo_view = serial.load(data_path)

    view_converter = DefaultViewConverter(topo_view.shape[1:])

    print 'converting data'
    train_X = view_converter.topo_view_to_design_mat(topo_view)

    warnings.warn('evaluating on train set')

    classify(fold_indices, train_X, train_y,  X = train_X, y = train_y,  **kwargs)


"""
def experiment(state, channel):
    Experiment script to launch jobman.

    Usage:
    jobman cmdline bilinear.scripts.basic.classify_bilinear.experiment TODO look at training script
    dataset = make(state.dataset)
    main(model_path = state.model_path,
         which_set = state.which_set,
         method = state.method,
         dataset = dataset)
    """


if __name__ == '__main__':
    """
    Useful for quick tests.
    Usage: python train_bilinear.py
    """

    parser = OptionParser()
    parser.add_option("-m", "--model",
                  action="store", type="string", dest="model",
                  help="path to pickled model file", metavar="MODEL",
                  default='model.pkl')
    parser.add_option("-d", "--data",
                action="store", type="string", dest="data")
    parser.add_option("--one-against-one", action="store_false", dest="one_against_many", default=True,
                      help="use a one-against-one classifier rather than a one-against-many classifier")


    (options, args) = parser.parse_args()
    assert os.path.exists(options.model)

    main(data_path=options.data,
         model_path=options.model,
         one_against_many = options.one_against_many,
    )
