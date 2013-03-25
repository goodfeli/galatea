from pylearn2.utils import serial
bow = serial.load('/data/lisatmp/goodfeli/final_bow.pkl')
words = bow.words
del bow
import gc
gc.collect
import os
labels_dir = '/data/lisatmp/goodfeli/esp/final_labels'
labels = sorted(os.listdir(labels_dir))

assert len(labels) == 1000

model = serial.load('rectifier_7_best.pkl')

from theano import tensor as T
from theano import function
X = T.matrix()
y = model.fprop(X)

f = function([X], y)

l2_path = '/data/lisatmp/goodfeli/esp/final_l2'
import numpy as np

from pylearn2.utils import image

imbase = '/data/lisatmp/goodfeli/esp/final_images'
ims = sorted(os.listdir(imbase))


for label, im in zip(labels, ims):

    stem = label.split('.')[0]
    assert stem in im

    img = image.load(imbase + '/' + im)

    image.show(img)

    full_label_path = labels_dir + '/' + label
    print 'True labels:'
    fd = open(full_label_path,'r')
    print fd.read()
    fd.close()

    full_l2_path = l2_path + '/' + label.split('.')[0] + '.npy'

    l2 = np.load(full_l2_path)

    y = f(l2)

    print 'Predicted labels: '
    print y.shape
    print [words[i] for i in xrange(len(words)) if y[0,i] >= 0.5]

    idxs = sorted(range(len(words)), key=lambda x:-y[0,x])

    print 'Top 10 labels:'
    print [words[idx] for idx in idxs[:10]]


    print 'Waiting...'
    x = raw_input()
    if x == 'q':
        quit()
    print 'Running...'
