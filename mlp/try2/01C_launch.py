#run with no args
import os
from pylearn2.utils.shell import run_shell_command

expdir = '/RQexec/goodfell/experiment_7'
names = os.listdir(expdir)
dirs = [expdir + '/' + name for name in names]
dirs = [d for d in dirs if not os.path.exists(d + '/cluster_info.txt')]

command = 'jobdispatch --torque --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True' + \
        ' --duree=48:00:00 --whitespace ' + \
        '--gpu bash /RQexec/goodfell/galatea/mlp/experiment_7/worker.sh '
command += ' "{{'

command += ', '.join(dirs)

command += '}}" '

print 'Running '+command
output, rc = run_shell_command(command)
print 'Output was'
print output

print 'Writing command output to all directories...'
for i, d in enumerate(dirs):
    out = d + '/cluster_info.txt'
    out = open(out, 'w')
    out.write("job id: "+str(i)+'\n')
    out.write(output)
    out.close()
