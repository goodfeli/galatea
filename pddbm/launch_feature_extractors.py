#run this with no arguments from the directory containing the yaml files. the npy files will get written here

import os

cwd = os.getcwd()

yamls = [ name for name in os.listdir('.') if name.endswith('.yaml') ]

assert len(yamls) > 0


command = 'jobdispatch --gpu --env=THEANO_FLAGS=device=gpu,force_device=True --duree=10:00:00 --whitespace python /RQusagers/goodfell/galatea/pddbm/extract_features.py '
command += cwd + '/'
command += '"{{'
command += ','.join(yamls)
command += '}}"'

os.system(command)


print command
