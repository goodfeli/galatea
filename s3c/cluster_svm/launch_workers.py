#python launch_workers.py training_set.npy

import sys
import os
from pylearn2.utils import serial

assert len(sys.argv) == 2
train_file = sys.argv[1]
assert train_file.endswith('.npy')

pieces = train_file.split('.npy')
assert len(pieces) == 2

results_dir = pieces[0]
serial.mkdir(results_dir)

command = 'jobdispatch --duree=48:00:00 --whitespace --mem=18G python /RQusagers/goodfell/galatea/s3c/cluster_svm/fold_point_worker.py --dataset cifar100 --train '
command += train_file
command += ' "{{'

options = []
for C in [.05,.1,.15,.2,.5,1]:
    for fold in xrange(5):
        options.append( '--fold %(fold)d -C %(C)f --out %(results_dir)s/%(fold)d_%(C)f.txt' % locals() )

command += ','.join(options)
command += '}}"'

os.system(command)


print command
