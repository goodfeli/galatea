from optparse import OptionParser
import warnings
try:
    from scikits.learn.metrics import classification_report
except ImportError:
    classification_report = None
    warnings.warn("couldn't find scikits.learn.metrics.classification_report")
try:
    from scikits.learn.metrics import confusion_matrix
except ImportError:
    confusion_matrix = None
    warnings.warn("couldn't find scikts.learn.metrics.confusion_matrix")
from galatea.s3c.feature_loading import get_features
from pylearn2.utils import serial
from pylearn2.datasets.cifar10 import CIFAR10
import numpy as np

def test(model, X, y, output_path):
    print "Evaluating svm"
    y_pred = model.predict(X)
    #try:
    if True:
        acc = (y == y_pred).mean()
        print "Accuracy ",acc
        f = open(output_path,'w')
        f.write('Accuracy: '+str(acc)+'\n')
        if classification_report:
            cr =  classification_report(y, y_pred)#, labels=selected_target,
                                #class_names=category_names[selected_target])
            print cr
            f.write(str(cr))
        if confusion_matrix:
            cm =  confusion_matrix(y, y_pred)#, labels=selected_target)
            print cm
            f.write(str(cm))
        f.close()
    """except:
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
        raise
"""
#


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


def main(model_path,
        test_path,
        output_path,
        dataset,
        split,
        **kwargs):

    model =  serial.load(model_path)

    cifar10 = dataset == 'cifar10'
    stl10 = dataset == 'stl10'
    assert cifar10 or stl10

    y = get_test_labels(cifar10, stl10)
    X = get_features(test_path, split)
    if stl10:
        num_examples = 8000
    if cifar10:
        num_examples = 10000
    assert X.shape[0] == num_examples
    assert y.shape[0] == num_examples

    test(model,X,y,output_path)


if __name__ == '__main__':
    """
    Useful for quick tests.
    Usage: python train_bilinear.py
    """

    parser = OptionParser()
    parser.add_option("-m", "--model",
                action="store", type="string", dest="model_path")
    parser.add_option("-t", "--test",
                action="store", type="string", dest="test")
    parser.add_option("--split", action="store_true", dest="split", default = False, help="double the example size by splitting each feature into a positive component and a negative component")
    parser.add_option("-o", action="store", dest="output", default = None, help="path to write the report to")
    parser.add_option('--dataset', type='string', dest = 'dataset', action='store', default = None)

    (options, args) = parser.parse_args()

    assert options.output

    main(model_path=options.model_path,
         test_path=options.test,
         output_path = options.output,
         dataset = options.dataset,
         split = options.split
    )
