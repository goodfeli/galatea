import os, sys, socket
from time import gmtime, strftime
from jobman import DD
from params_default import default_config
import numpy

class Jobmaster:
    def __init__(self):
        """
        a jobdispatch command: header + tail
        """
        self.header = 'jobman -r cmdline main.main '
        # this file saves all jobdispatch commands
        self.jobdispatch_file = 'jobdispatch_'+ self._get_timestamp() + '.sh'
        self.default_config = default_config
        
        # commands come after header
        self.tails = []
        self.jobmans = []
        self.jobs= []

        #-------------------------------------------------------------------------
        # put the hyper_parameters that we want to change for different jobs here
        self.l2 = [0,1,0.1,0.01,0.001,0.0001,0.00001,0.000001]
        #-------------------------------------------------------------------------
    def _get_timestamp(self):
        # the name of the jobman folder
        return strftime("%a-%d-%b-%Y-%H-%M-%S", gmtime())

    def _compose_one_job(self, args):
        cmd = ''
        for (k, v) in sorted(args.iteritems()):
            cmd += k + '=' + str(v) + ' '
        return self.header + cmd

    def _set_tails(self):
        config = self.default_config.copy()
        
        for i in self.l2:
            #config['gbs'] = i
            config['l2norm'] = i
            #config['ebs'] = i
            #config['cbs'] = i
            self.tails.append(config.copy())
            
    def write_job_file(self):
        # write tails
        self._set_tails()
        
        for i in self.tails:
            self.jobmans.append(self._compose_one_job(i))
            
        
    def write_jobdispatch_cmd_file(self):    
        
        hostname = socket.gethostname()
        header = '#!/bin/bash\n'
        #content = 'jobdispatch --mem=70000 --file=%s' % self.job_file
        content = 'jobdispatch --gpu --mem=6000 '
        if hostname == 'ip05':
            # Mammouth cluster.
            content += '--bqtools --duree=100000 '
        elif hostname.startswith('colosse'):
            # Colosse cluster.
            content += '--sge --duree=100000 --queue=short '
        elif hostname.startswith('briaree'):
            # Briaree cluster.
            pass
        elif hostname.startswith('mon'):
            # Monk with gpu on sharcnet, maybe do something
            pass
        else:
            # Assuming LISA cluster.
            content += '--condor '

        for job in self.jobmans:
            self.jobs.append(content + job) 

        to_write = '\n\n'.join(self.jobs)
        f = open(self.jobdispatch_file, 'w')
        f.write(header + to_write)
        f.close()

if __name__ == '__main__':
    master = Jobmaster()
    master.write_job_file()
    master.write_jobdispatch_cmd_file()
