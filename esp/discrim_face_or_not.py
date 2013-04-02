import numpy as np

x = raw_input('use esp game dataset?')

if x == 'y':
    from galatea.esp import Im2Word
    dataset = Im2Word(start=99000, stop=100000
                          )
else:
    from galatea.esp import FinalIm2Word
    dataset = FinalIm2Word()

x = raw_input('use people?')


people_mask = np.zeros((dataset.y.shape[0],), dtype='bool')

for word in ['face', 'person', 'people', 'man', 'men', 'woman', 'women',
        'child', 'children', 'kid', 'kids', 'guy', 'boy', 'boys', 'girl',
        'girls', 'infant', 'infants', 'baby', 'babies']:
    if word not in dataset.words:
        continue
    idx = dataset.words.index(word)
    mask = dataset.y[:,idx].astype('bool')
    people_mask = people_mask | mask

if x == 'y':
    mask = people_mask
else:
    assert x == 'n'
    mask = (1- people_mask).astype('bool')

num_feat = dataset.X.shape[1]
dataset.X = dataset.X[mask, :]
assert dataset.X.shape == (mask.sum(), num_feat)

dataset.y = dataset.y[mask, :]
assert dataset.y.shape == (mask.sum(), len(dataset.words))


from pylearn2.utils import serial

model = serial.load('rectifier_7.pkl')

import theano.tensor as T
X = T.matrix()
state = model.fprop(X)
target = T.matrix()

right_cost = model.layers[-1].kl(Y=target, Y_hat=state)
wrong_cost = model.layers[-1].kl(Y=target[::-1,:], Y_hat=state)

#from theano.printing import Print
#right_cost = Print('right_cost')(right_cost)

acc = (wrong_cost > right_cost).mean()

from theano import function

f = function([X, target], acc)

acc = f(dataset.X, dataset.y)

print acc
