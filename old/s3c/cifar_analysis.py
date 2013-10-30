coarse_classes_in_tlc = [3, 4, 6, 7, 11, 12]

import numpy as np

from pylearn2.utils.serial import load

cifar100_train = load('${PYLEARN2_DATA_PATH}/cifar100/cifar-100-python/train')
cifar100_meta = load('${PYLEARN2_DATA_PATH}/cifar100/cifar-100-python/meta')

fine_labels = np.asarray(cifar100_train['fine_labels'])
assert fine_labels.max() == 99
coarse_labels = np.asarray(cifar100_train['coarse_labels'])

print cifar100_meta.keys()

coarse_label_names = cifar100_meta['coarse_label_names']
fine_label_names = cifar100_meta['fine_label_names']

for coarse_class in coarse_classes_in_tlc:
    coarse_name = coarse_label_names[coarse_class]

    print coarse_name,' (',coarse_class,')'

    fine_label_instances = fine_labels[coarse_labels == coarse_class ]

    for label in xrange(100):
        if label in fine_label_instances:
            print '\t',fine_label_names[label],' (',label,')'




