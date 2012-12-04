#run with no args
import os
num_jobs = 50


command = 'jobdispatch --torque --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=48:00:00 --whitespace --gpu train.py '
command += ' "{{'

files = [ '/RQexec/goodfell/experiment_3/%d/stage_00_inpaint.yaml ' %idx for idx in xrange(num_jobs) ]

command += ', '.join(files)

command += '}}" '

os.system(command)


print command
