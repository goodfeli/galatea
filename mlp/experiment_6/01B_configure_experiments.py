import os
from pylearn2.utils.serial import mkdir
from pylearn2.config import yaml_parse
from pylearn2.utils import safe_zip
import shutil

results = open("results.dat", "r")
lines = results.readlines()
results.close()

params = yaml_parse.load_path('params.yaml')

validate = open('validate.yaml', 'r')
validate_template = validate.read()
validate.close()

for expnum, line in enumerate(lines):
    elems = line.split(' ')
    assert elems[-1] == '\n'
    obj = elems[0]
    if obj == 'P':
        expdir = '/RQexec/goodfell/experiment_6/%d' % expnum
        if os.path.exists(expdir):
            continue
        try:
            mkdir(expdir)

            config = {}
            for param, value in safe_zip(params, elems[2:-1]):
                if param['type'] == 'float':
                    value = float(value)
                elif param['type'] == 'int':
                    value = int(value)
                else:
                    raise NotImplementedError()
                if 'postprocess' in param:
                    value = param['postprocess'](value)
                if 'joint_postprocess' in param:
                    try:
                        value = param['joint_postprocess'](value, config)
                    except Exception, e:
                        print 'Exception: ',e
                        raise
                config[param['name']] = value

            validate = open(expdir + '/validate.yaml', 'w')
            validate.write(validate_template % config)
            validate.close()
        except:
            shutil.rmtree(expdir)
    else:
        try:
            obj = float(obj)
        except:
            print "Something is wrong with line %d, its objective value is listed as " % expnum, obj
