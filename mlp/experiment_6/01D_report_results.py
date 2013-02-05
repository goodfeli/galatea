import os
import shutil

from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from pylearn2.utils import shell


status, rc = shell.run_shell_command("qstat -u goodfell -t @hades")
assert rc == 0

results = open("results.dat", "r")
lines = results.readlines()
results.close()

params = yaml_parse.load_path('params.yaml')
added = 0
print 'Experiment numbers reported by this script start at 0.'
print 'Keep in mind that vim will refer to the first line of results.dat as line 1'
for expnum, line in enumerate(lines):
    elems = line.split(' ')
    assert elems[-1] == '\n'
    obj = elems[0]
    if obj == 'P':
        # print expnum, 'pending according to results.dat'
        expdir = '/RQexec/goodfell/experiment_6/%d' % expnum
        if not os.path.exists(expdir):
            print 'Experiment not yet configured for experiment',expnum
            continue
        cluster_info = expdir + '/cluster_info.txt'
        if not os.path.exists(cluster_info):
            print 'Experiment not yet launched for experiment',expnum
            continue

        if os.path.exists(expdir + '/infeasible.txt'):
            print '\tinfeasible'
            obj = 1.
            time = 200000.
            elems[0] = str(obj)
            elems[1] = str(time)
            line = ' '.join(elems)
            lines[expnum] = line
            added += 1
            continue
        if os.path.exists(expdir + '/bugged.txt'):
            print expnum,'must be re-run after fixing a bug'
            continue


        f = open(cluster_info, 'r')
        flines = f.readlines()
        f.close()
        l = flines[0]
        chunks = l.split('job id:')
        job_id = int(chunks[-1][0:-1])
        array_id = flines[1][:-1]
        assert array_id.endswith('[].hades')
        array_num, empty = array_id.split('[].hades')
        array_num = int(array_num) # make sure this is really an integer
        assert empty == ''
        full_job_id = '%d[%d].hades' % (array_num, job_id)

        if full_job_id in status:
            # print '\tqstat confirms job still pending'
            continue


        found = False
        for line in flines:
            start = "[DBI] All logs will be in the directory:  "
            if line.startswith(start):
                line = line.split(start)
                log_dir = line[-1][0:-1]
                found = True
                break
        assert found
        log_files = os.listdir(log_dir)

        found = False
        for f in log_files:
            if f.endswith('err-%d' % job_id):
                found = True
                break
        if found:
            f = log_dir + '/' + f
            #print 'error file:', f
        else:
            f = '<not found>'
            #print 'Could not find log file for experiment',expnum

        if os.path.exists(expdir+'/fail.txt'):
            message = str(expnum)+' failed!!!!'
            new_f = expdir + '/log.err'
            shutil.copyfile(f, new_f)
            shutil.copyfile(f.replace('err','out'),new_f.replace('err','out'))
            message += 'Log file: ' + new_f
            print message
            continue


        if os.path.exists(expdir + '/new_protocol.txt'):
            valid_results = expdir + '/validate_best.txt'
            if os.path.exists(valid_results):
                f = open(valid_results)
                l = f.readlines()
                f.close()
                valid_err_str = "valid_y_misclass : "
                time_str = "time trained:  "
                found_obj = False
                found_time = False
                for line in l:
                    if line.startswith(time_str):
                        time = float(line[len(time_str):-1])
                        found_time = True
                    if line.startswith(valid_err_str):
                        obj = float(line[len(valid_err_str):-1])
                        found_obj = True
                assert found_obj
                assert found_time
            else:
                print '\tvalidate_best.txt never showed up'
                print '\tlog file: ',f
                continue
        else:
            if not os.path.exists(expdir+'/validate_best.pkl'):
                print '\tExperiment done running but validate_best.pkl never showed up'
                print '\tLog file:',f
                print '\tValidation not yet run for experiment',expnum
                continue
            # we have to check if this exists, then load it
            # if we load it and respond to the exception, the exception is not specific enough
            # to know that the problem is the path not existing. if we check if the path exists
            # after we get the exception, the filesystem could have caught up while we're doing
            # the exception handling, and we'd decide that the exception must indicate a real
            # problem
            if not os.path.exists(expdir+'/validate.pkl'):
                print 'Experiment',expnum,' is suffering from the filesystem being stupid... validate_best.pkl exists but validate.pkl does not'
                continue

            model = serial.load(expdir+'/validate.pkl')

            monitor = model.monitor

            if not monitor.training_succeeded:
                print 'Training not done yet for experiment',expnum
                continue
            time = monitor.channels['valid_y_misclass'].time_record[-1]

            model = serial.load(expdir+'/validate_best.pkl')
            monitor = model.monitor

            obj = monitor.channels['valid_y_misclass'].val_record[-1]
            assert obj == min(monitor.channels['valid_y_misclass'].val_record)

        print expnum,obj,time

        elems[0] = str(obj)
        elems[1] = str(time)
        line = ' '.join(elems)
        lines[expnum] = line
        added += 1
    else:
        try:
            obj = float(obj)
        except:
            print "Something is wrong with experiment %d, its objective value is listed as " % expnum, obj

out = ''.join(lines)
of = open('results.dat', 'w')
of.write(out)
of.close()

print added,'results added'
