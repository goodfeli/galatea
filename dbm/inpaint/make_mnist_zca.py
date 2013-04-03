from pylearn2.datasets.mnist import MNIST
train = MNIST(which_set = 'train')
from pylearn2.datasets.preprocessing import ZCA
zca = ZCA()
zca.apply(train, can_fit=True)
from pylearn2.utils import serial
serial.save('mnist_zca.pkl', zca)
