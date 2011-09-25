from optparse import OptionParser
from galatea.s3c.feature_loading import get_features
from pylearn2.utils import serial
from pylearn2.gui.patch_viewer import make_viewer


def get_test_data():
    print 'loading entire stl-10 dataset'
    stl10 = serial.load("${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/test.pkl")
    return stl10


def main(model_path,
        data_path,
        split,
        **kwargs):

    model =  serial.load(model_path)

    raw_dataset = get_test_data()
    X = get_features(data_path, split)
    assert X.shape[0] == 8000

    size = 100
    for start in xrange(0,X.shape[0]-size,size):
        raw_X = raw_dataset.X[start:start+size,:]
        pv = make_viewer(raw_dataset.X / 127.5, rescale = False)
        pv.show()

        y = raw_dataset.y[start:start+size]
        pred_y = model.predict(X[start:start+size,:])
        for i in xrange(y.shape[0]):
            if y[i] == pred_y[i]:
                print str(start+i)+': correct ('+raw_dataset.class_names[y[i]]+')'
            else:
                print str(start+i)+': mistook '+raw_dataset.class_names[y[i]]+' for '+raw_dataset.class_names[pred_y[i]]
        x = raw_input()
        if x == 'q':
            break

if __name__ == '__main__':
    """
    Useful for quick tests.
    Usage: python train_bilinear.py
    """

    parser = OptionParser()
    parser.add_option("-m", "--model",
                action="store", type="string", dest="model_path")
    parser.add_option("-d", "--data",
                action="store", type="string", dest="data")
    parser.add_option("--split", action="store_true", dest="split", default = False, help="double the example size by splitting each feature into a positive component and a negative component")


    (options, args) = parser.parse_args()

    main(model_path=options.model_path,
         data_path=options.test,
         split = options.split
    )
