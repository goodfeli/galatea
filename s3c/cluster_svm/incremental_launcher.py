import subprocess as sp
import time

def get_count():
    command = "qstat -u goodfell -t"

    process = sp.Popen(command, shell = True, stdout = sp.PIPE, stderr = sp.PIPE)

    output, error = process.communicate()

    return len([0 for x in output.split('\n') if x.find('courte') != -1])




args_to_run = [ 'omp_2_train_0_5.mat',
                'omp_2_train_1.mat',
                'omp_2_train_0_25.mat',
                'omp_2_train_0_1.mat',
                'sc_3_800_train_1_5.mat',
                'sc_3_800_train_1.mat',
                'sc_3_800_train_0_75.mat',
                'sc_3_800_train_0_5.npy.mat' ]


for arg in args_to_run:
    print 'preparing to launch',arg

    while True:
        count = get_count()

        if count + 25 <= 72:
            break
        else:
            print 'There are currently %d jobs, waiting' % count
            time.sleep(300)

    command = "python launch_workers_matlab.py /RQexec/goodfell/"+arg

    process = sp.Popen(command, shell = True, stdout = sp.PIPE, stderr = sp.PIPE)

    output, error = process.communicate()

    print 'output: '
    print output
    print 'errors: '
    print error
