from pylearn2.utils import serial
bow = serial.load('/data/lisatmp/goodfeli/final_bow.pkl')
words = bow.words
import os
labels_dir = '/data/lisatmp/goodfeli/esp/final_labels'
labels = sorted(os.listdir(labels_dir))


assert len(labels) == 1000

l2_path = '/data/lisatmp/goodfeli/esp/final_l2'
import numpy as np

from pylearn2.utils import image

imbase = '/data/lisatmp/goodfeli/esp/final_images'
ims = sorted(os.listdir(imbase))


right = 0
total = 0

rng = np.random.RandomState([1,2,3])

for idx, packed in enumerate(zip(labels, ims)):
    true_label, im = packed

    dists = np.square(bow.X - bow.X[idx, :]).sum(axis=1)
    dists[idx] = np.inf
    wrong_idx = np.argmin(dists)
    assert wrong_idx != idx
    print 'wrong_idx ',wrong_idx
    wrong_label = labels[wrong_idx]


    true_stem = true_label.split('.')[0]
    assert true_stem in im

    img = image.load(imbase + '/' + im)

    image.show(img)

    true_idx = rng.randint(2)


    if true_idx == 0:
        guess_labels = [true_label, wrong_label]
    else:
        assert true_idx == 1
        guess_labels = [wrong_label, true_label]


    for i, label in enumerate(guess_labels):
        print 'label',i
        full_label_path = labels_dir + '/' + label
        fd = open(full_label_path,'r')
        print fd.read()
        fd.close()
        print
    x = ''
    while x not in ['0', '1', 'q']:
        x = raw_input()
    if x == 'q':
        quit()
    total += 1
    right += int(x) == true_idx
    print float(right) / float(total)

