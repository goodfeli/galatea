#python launch_workers.py training_set.npy <comma separated list of C values> <mem req'd>
#e.g.
# launch_workers.py foo.npy 100.,1000.,10000. 12G
# launch_workers.py bar.npy

print 'python launched'
import sys
import os
from pylearn2.utils import serial
print 'done with imports'

assert len(sys.argv) in [2,3,4]
train_file = sys.argv[1]

if train_file.endswith('.npy'):
    pieces = train_file.split('.npy')
elif train_file.endswith('mat'):
    pieces = train_file.split('.mat')
else:
    assert False
assert len(pieces) == 2

results_dir = pieces[0]
serial.mkdir(results_dir)

if len(sys.argv) > 3:
    memreq = sys.argv[3]
else:
    memreq = '12G'

command = 'jobdispatch --duree=48:00:00 --whitespace --mem=%(memreq)s /RQusagers/goodfell/cifar10_fold_point_worker ' % locals()
command += ' "{{'

if len(sys.argv) > 2:

    C_list = sys.argv[2]

    C_list = [ float(C) for C in C_list.split(',') ]

else:
    C_list = [ .1, 1., 10., 100., 1000., 1e4 ]


options = []
for C in C_list:
    for fold in xrange(5):
        options.append( '%(fold)d %(C)f' % locals() )

command += ','.join(options)
command += '}}" %(train_file)s %(results_dir)s' % locals()

print 'running command'
os.system(command)
print 'done running command, command was:'

print command
