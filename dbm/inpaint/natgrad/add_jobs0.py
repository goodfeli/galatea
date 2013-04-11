#!/usr/bin/python

# Razvan Pascanu
'''
  Script to add more targeted jobs into the db
'''

from jobman.tools import DD
from jobman import sql
import numpy

# import types of architectures we want to use


if __name__=='__main__':

    TABLE_NAME='nips_2013_test_inpainting001'

    db = sql.db('postgres://pascanur:he1enush@opter.iro.umontreal.ca/pascanur_db/'+TABLE_NAME)
    #import ipdb; ipdb.set_trace()
    #db = sql.postgres_serial( user = 'pascanur',
    #        password='he1enush',
    #        host='gershwin',
    #        port = 5432,
    #        database='pascanur_db',
    #        table_prefix = TABLE_NAME)

    state = DD()

    ## DEFAULT VALUES ##
    state['jobman.experiment'] = 'run_dbm_inpainting.jobman'
    state['path'] = '/RQexec/pascanur/data/mnist.npz'
    state['samebatch'] = 1
    state['mbs'] = 128*2
    state['bs']  = 128*2
    state['ebs'] = 128*2
    state['cbs'] = 128*2

    state['loopIters'] = 6000
    state['timeStop'] = 45*60
    state['minerr'] = 1e-5

    state['lr'] = .2
    state['lr_adapt'] = 0

    state['damp'] = 5.
    state['adapt'] = 1.
    state['mindamp'] = .5
    state['damp_ratio'] =5./4.
    state['minerr'] = 1e-5

    state['seed'] = 123


    state['profile'] = 0
    state['minresQLP'] = 1
    state['mrtol'] = 1e-4
    state['miters'] = 100
    state['trancond'] = 1e-4


    state['trainFreq'] = 1
    state['validFreq'] = 2000
    state['saveFreq'] = 30

    state['prefix'] = 'inpaint_'
    state['overwrite'] = 0
    n_jobs = 0
    '''
    # Fast small
    state['miters'] = 40
    state['lr'] = .1
    state['bs'] = 100
    state['mbs'] = 100
    state['cbs'] = 100
    state['ebs'] = 100
    state['loopIters'] = int(5e6)
    state['trainFreq'] = 20
    sql.add_experiments_to_db(
        [state], db, verbose=1, force_dup =True)
    n_jobs += 1

    state['lr'] = .02
    state['bs'] = 100
    state['mbs'] = 100
    state['cbs'] = 100
    state['ebs'] = 100
    state['loopIters'] = int(5e6)
    state['trainFreq'] = 20
    sql.add_experiments_to_db(
        [state], db, verbose=1, force_dup =True)
    n_jobs += 1


    # Large slow
    state['miters'] = 100
    state['lr'] = .1
    state['bs'] = 250
    state['mbs'] = 250
    state['cbs'] = 250
    state['ebs'] = 250
    state['loopIters'] = int(5e6)
    state['trainFreq'] = 20
    sql.add_experiments_to_db(
        [state], db, verbose=1, force_dup =True)
    n_jobs += 1

    state['miters'] = 100
    state['mindamp'] = .02
    state['lr'] = .1
    state['bs'] = 250
    state['mbs'] = 250
    state['cbs'] = 250
    state['ebs'] = 250
    state['loopIters'] = int(5e6)
    state['trainFreq'] = 20
    sql.add_experiments_to_db(
        [state], db, verbose=1, force_dup =True)
    n_jobs += 1
    '''
    state['miters'] = 100
    state['lr'] = .5
    state['bs'] = 250
    state['mbs'] = 250
    state['cbs'] = 250
    state['ebs'] = 250
    state['loopIters'] = int(5e6)
    state['trainFreq'] = 20
    sql.add_experiments_to_db(
        [state], db, verbose=1, force_dup =True)
    n_jobs += 1

    state['miters'] = 100
    state['lr'] = 1.
    state['bs'] = 250
    state['mbs'] = 250
    state['cbs'] = 250
    state['ebs'] = 250
    state['loopIters'] = int(5e6)
    state['trainFreq'] = 20
    sql.add_experiments_to_db(
        [state], db, verbose=1, force_dup =True)
    n_jobs += 1

    state['mindamp'] = .01
    state['miters'] = 100
    state['lr'] = .5
    state['bs'] = 250
    state['mbs'] = 250
    state['cbs'] = 250
    state['ebs'] = 250
    state['loopIters'] = int(5e6)
    state['trainFreq'] = 20
    sql.add_experiments_to_db(
        [state], db, verbose=1, force_dup =True)
    n_jobs += 1

    state['miters'] = 100
    state['lr'] = 1.
    state['bs'] = 250
    state['mbs'] = 250
    state['cbs'] = 250
    state['ebs'] = 250
    state['loopIters'] = int(5e6)
    state['trainFreq'] = 20
    sql.add_experiments_to_db(
        [state], db, verbose=1, force_dup =True)
    n_jobs += 1
    print 'N_jobs ', n_jobs, TABLE_NAME
