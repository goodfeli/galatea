from pylearn2.utils import serial
from pylearn2.datasets.cifar10 import CIFAR10
import numpy as np
import sys
from pylearn2.models.svm import DenseMulticlassSVM

print 'Loading labels'
y = CIFAR10(which_set = 'train', one_hot = True).y

ignore, features_path = sys.argv

print 'Loading features'
X = serial.load(features_path)
X = X.astype('float64') # avoid duplicating memory in sklearn

def train(X,y):
    print 'Training SVM...'
    return DenseMulticlassSVM(C =1., kernel='linear').fit(X, np.argmax(y, axis=1))

X_train = X[0:40000,:]
y_train = y[0:40000,:]

W = train(X_train, y_train)

def acc(W,X,y):
    y_hat = W.predict()
    y = np.argmax(y, axis=1)
    acc = (y_hat == y).mean()
    return acc

print 'Train acc',acc(W,X_train,y_train)

X_valid = X[40000:,:]
y_valid = y[40000:,:]

print 'Valid acc',acc(W,X_valid,y_valid)
