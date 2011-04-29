from framework.utils import serial
from framework.datasets import mnist
from framework.datasets import preprocessing
import numpy as N

train = mnist.MNIST(which_set="train")

pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ZCA(n_drop_components = 67))

test = mnist.MNIST(which_set="test")

train.apply_preprocessor(preprocessor = pipeline, can_fit = True)
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)


rng = N.random.RandomState([1,2,3])

bad = train.X.std(axis=0) == 0

def fix(dataset):
    dataset.X[:,bad] = rng.randn(dataset.X.shape[0],bad.sum()) * .3

fix(train)
fix(test)

train.enable_compression()
test.enable_compression()

print train.X.mean(axis=0)

serial.save('mnist_preprocessed_train.pkl',train)
serial.save('mnist_preprocessed_test.pkl',test)


