#Throwaway script added by Ian Goodfellow

import os

base = '/data/lisatmp/data/esp_game/ESPGame100k/labels/'
paths = sorted(os.listdir(base))
assert len(paths)==100000

words = {}

for i, path in enumerate(paths):

    if i % 1000 == 0:
        print i
    path = base+path
    f = open(path,'r')
    lines = f.readlines()
    for line in lines:
        word = line[:-1]
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1

ranked_words = sorted(words.keys(), key=lambda x: -words[x])

ranked_words = ranked_words[0:4000]

import numpy as np
X = np.zeros((100000, 4000))

words = set(ranked_words)

for i, path in enumerate(paths):

    if i % 1000 == 0:
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
dataset.use_design_loc('/data/lisatmp/goodfeli/esp_bow.npy')
assert dataset.X.min() == 0.
assert dataset.X.max() == 1.
from pylearn2.utils import serial
serial.save('/data/lisatmp/goodfeli/esp_bow.pkl', dataset)

