import os, sys, socket
from time import gmtime, strftime
from jobman import DD
from params_default import default_config
import numpy

class Jobmaster:
    def __init__(self):
        self.header = 'jobman -r cmdline run_autoencoder_jobman.main '
        self.job_file = 'jobman_'+ self._get_timestamp() + '_test.tmp'
        self.jobdispatch_file = 'jobdispatch_'+ self._get_timestamp() + '.sh'
        self.default_config = default_config
        # commands come after header
        self.tails = []
        self.jobmans = []
        self.jobs= []
        """
        self.lr = [0.03]
        self.n_hidden = [1, 5, 10, 15, 20, 50, 100, 200, 500, 1000]
        self.ratio_pattern = [0.005]  
        self.L1 = [0.0001]
        self.corruption = [0.1]
        """
        #self.dataset = ['mnist', 'curves']
        #self.preconditioner = ['martens', 'jacobi']
        self.minibatch_size = [5000,2000,500]
    def _get_timestamp(self):
        return strftime("%a-%d-%b-%Y-%H-%M-%S", gmtime())

    def _compose_one_job(self, args):
        cmd = ''
        for (k, v) in sorted(args.iteritems()):
            cmd += k + '=' + str(v) + ' '
        return self.header + cmd

    def _set_tails(self):
        config = self.default_config.copy()
        
        #for i in self.dataset:
            #for j in self.n_hidden:
            #    for k in self.ratio_pattern:
            #        for l in self.L1:
            #            for m in self.corruption:
            #                config['lr'] = i
            #                config['n_hidden'] = j
            #                config['ratio_pattern'] = k
            #                config['L1'] = l
            #                config['corruption'] = m
            #                self.tails.append(config.copy())
            #config['dataset'] = i
        for i in self.minibatch_size:
            config['global_pretraining_batch_size'] = i
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
        content = 'jobdispatch --gpu --mem=8000 '
        if hostname == 'ip05':
            # Mammouth cluster.
            content += '--bqtools --duree=10000 '
        elif hostname.startswith('colosse'):
            # Colosse cluster.
            content += '--sge --duree=10000 --queue=short '
        elif hostname.startswith('briaree'):
            # Briaree cluster.
            pass
        elif hostname.startswith('mon'):
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
        
