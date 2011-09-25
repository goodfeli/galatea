from pylearn2.utils import serial

train = serial.load('${PYLEARN2_DATA_PATH}/stl10/stl10_32x32/train.pkl').X
print train.mean()
