#run with no args
import os
num_jobs = 100


command = 'jobdispatch --torque --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=48:00:00 --whitespace --gpu bash '
command += ' "{{'

files = [ '/RQexec/goodfell/experiment_4/%d/pipeline.sh ' %idx for idx in xrange(num_jobs) ]

command += ', '.join(files)

command += '}}" '

os.system(command)


print command
