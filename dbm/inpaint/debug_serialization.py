from pylearn2.config import yaml_parse
from pylearn2.utils import environ

environ.putenv('PYLEARN2_TRAIN_FILE_FULL_STEM','expdir/cifar10_M9')

train = yaml_parse.load_path('expdir/cifar10_M9.yaml')

dataset = train.dataset

def raise_assert():
    assert False

dataset.__getstate__ = raise_assert

from pylearn2.utils import serial

serial.to_string(train.model)

train.algorithm.setup(train.model, dataset)

serial.to_string(train.model)
train.model.monitor()
serial.to_string(train.model.monitor)
serial.to_string(train.model)
train.algorithm.train(dataset)
serial.to_string(train.model.monitor)
serial.to_string(train.model)
