#Throwaway script added by Ian Goodfellow

import os

train = '/data/lisatmp/goodfeli/esp_bow.pkl'

from pylearn2.utils import serial

train = serial.load(train)
del train.X

base = '/data/lisatmp/goodfeli/esp/final_labels/'
paths = sorted(os.listdir(base))
assert len(paths)==1000

ranked_words = train.words
words = set(ranked_words)

import numpy as np
X = np.zeros((1000, 4000))

for i, path in enumerate(paths):

    if i % 100 == 0:
        print i
    path = base+path
    f = open(path,'r')
    lines = f.readlines()
    for line in lines:
        word = line[:-1]
        if word in words:
            X[i, ranked_words.index(word)] = 1

from galatea.esp.bow import BagOfWords
dataset = BagOfWords(X=X, words=ranked_words, files=paths)
dataset.use_design_loc('/data/lisatmp/goodfeli/final_bow.npy')
assert dataset.X.min() == 0.
assert dataset.X.max() == 1.
serial.save('/data/lisatmp/goodfeli/final_bow.pkl', dataset)

