#!/usr/bin/python

# Razvan Pascanu
'''
  Script to add more targeted jobs into the db
'''

from jobman.tools import DD
from jobman import sql

# import types of architectures we want to use
import daa
import main
import mlp
import numpy


if __name__=='__main__':

    TABLE_NAME='dragonball_cifarmlp001'

    db = sql.db('postgres://pascanur:he1enush@gershwin.iro.umontreal.ca/pascanur_db/'+TABLE_NAME)
    #db = sql.postgres_serial( user = 'pascanur',
    #        password='he1enush',
    #        host='colosse1',
    #        port = 5432,
    #        database='pascanur_db',
    #        table_prefix = TABLE_NAME)

    state = DD()

    ## DEFAULT VALUES ##
    state['jobman.experiment'] = 'main.main'
    n_jobs = 0
    state['data'] = ('/home/pascanur/data/' +
                       'mnist.npz')
    state['profile'] = 1
    state['verbose'] = 3
    state['device'] = 'gpu'
    state['gbs'] = 50000
    state['mbs'] = 25000
    state['ebs'] = 25000
    state['cbs'] = 200
    state['model'] = 'mlp'
    state['algo'] = 'natSGD_jacobi'
    state['type'] = 'Gf'
    state['loopIters'] = 400
    state['gotNaN'] = 0
    state['seed'] = 913
    state['daacost'] = 'cross'
    state['hids'] =  '[200,200,200]'
    state['mrtol'] = 1e-4
    state['mreg'] = 0
    state['jreg'] = .02
    state['resetFreq'] = 40
    state['miters'] = numpy.int32(20)
    state['lr'] = .01
    state['lsIters'] = 80
    state['checkFreq'] = 5
    state['krylovDim'] = 15
    state['lbfgsIters'] = 30


    state['data'] = '/home/pascanur/data/cifar10.npz'
    state['mreg'] = 1.
    state['adaptivedamp'] = 1
    state['model'] = 'mlp'
    state['metric'] = 'Gf'
    state['algo'] = 'natNCG'
    state['gbs'] = 40000
    state['mbs'] = 5000
    state['ebs'] = 5000
    state['init'] = 'xavier'
    for hids in [
            '[2000,1000,1000]',
            '[3000,1000,1000]',
            '[2000,1000,1000,1000]',
            '[3000,1500,1000,500]',
            '[2000,1000,1000,1000,500,500]']:
        state['hids'] = hids
        n_jobs += 1
        sql.add_experiments_to_db([state], db, verbose=1, force_dup = True)


    print 'N_jobs ', n_jobs, TABLE_NAME

