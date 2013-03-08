from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.tfd import TFD


train = TFD(which_set = 'train')


preprocessor = preprocessing.Pipeline()
preprocessor.items.append(preprocessing.GlobalContrastNormalization())
preprocessor.items.append(preprocessing.ZCA())

preprocessor.apply(train, can_fit=True)

serial.save('tfd_gcn_whitener.pkl',preprocessor)

