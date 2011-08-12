job_name = 'cfa_olshausen'
chunk_size = 500
batch_size = 10

from pylearn2.utils import serial
import SkyNet


SkyNet.set_job_name(job_name)
components = SkyNet.get_dir_path('components')

num_examples = serial.load(components+'/num_examples.pkl')
serial.save(components+'/chunk_size.pkl',chunk_size)
serial.save(components+'/batch_size.pkl',batch_size)

command = '--mem=2000 '+SkyNet.get_user_dir(tmp = False, force = SkyNet.cluster) + '/galatea/contractive_feature_analysis/mnist_experiment_condor/stage_3_worker.py '+job_name+' "{{'

assert num_examples % chunk_size == 0

for b in xrange(0,num_examples,chunk_size):
    if b != 0:
        command+=','
    command += str(b)
command += '}}"'

SkyNet.launch_job(command)
