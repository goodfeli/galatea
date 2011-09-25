from optparse import OptionParser
from scikits.learn.metrics import classification_report
from scikits.learn.metrics import confusion_matrix
from galatea.s3c.feature_loading import get_features
from pylearn2.utils import serial

def test(model, X, y, output_path):
    print "Evaluating svm"
    y_pred = model.predict(X)
    try:
        acc = (y == y_pred).mean()
        print "Accuracy ",acc
        cr =  classification_report(y, y_pred)#, labels=selected_target,
                                #class_names=category_names[selected_target])

        cm =  confusion_matrix(y, y_pred)#, labels=selected_target)
        print cr
        print cm
        f = open(output_path,'w')
        f.write('Accuracy: '+str(acc)+'\n')
        f.write(str(cr))
        f.write(str(cm))
        f.close()
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
        raise
#


def get_test_labels():
    print 'loading entire stl-10 test set just to get the labels'
    stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/test.pkl")
    return stl10.y


def main(model_path,
        test_path,
        output_path,
        split,
        **kwargs):

    model =  serial.load(model_path)

    y = get_test_labels()
    X = get_features(test_path, split)
    assert X.shape[0] == 8000

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

    (options, args) = parser.parse_args()

    assert options.output

    main(model_path=options.model_path,
         test_path=options.test,
         output_path = options.output,
         split = options.split
    )
