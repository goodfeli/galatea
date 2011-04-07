from jobman import sql, DD, flatten, expand
from jobman.parse import filemerge

from experiments.avicenna_pca_da.scripts import train_pca_da

def update_view(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    # user-friendly view
    db.createView(TABLE_NAME + 'view')

def first_xp(TABLE_NAME):
  
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    cnt =0
    state = DD()
    prefix = '/data/lisatmp/ift6266h11/ryane/' + TABLE_NAME + '/'


    state.epochs = 0
    state.batch_size = 20
    state.solution = ''
    state.sparse_penalty = 0
    state.sparsityTarget = 0
    state.sparsityTargetPenalty = 0

    for training_epochs in [10,25,50]:
        for penalty_solution in ['l1_penalty', 'sqr_penalty']:
                state.epochs = training_epochs
                state.solution = penalty_solution
                
                if(penalty_solution == 'l1_penalty') :
                    for penalty in [1e-10, 1e-3, 0.01 , 0.05 , 0.1 , 0.5 , 0 , 1 , 2]:
                        state.sparse_penalty = penalty
                        cnt += 1
                        state.savedir = prefix + str(cnt) + '/'
                        sql.insert_job(train_pca_da, flatten(state), db)
                                                        
                
                elif(penalty_solution == 'sqr_penalty'):
                    for spTarPenal in [1e-10, 1e-4, 1e-3, 0.01 , 0.05 , 0.1 , 0.5 , 0 , 1 ]:
                        for spTar in [0.001, 0.01, 0.02, 0.03, 0.04, 0.05 , 0.06 , 1 , 2]:
                            state.sparsityTarget = spTar
                            state.sparsityTargetPenalty = spTarPenal
                            
                            cnt += 1
                            state.savedir = prefix + str(cnt) + '/'
                            sql.insert_job(train_pca_da, flatten(state), db)
                                                                            
                                  
                 
          
    update_view(TABLE_NAME)
    print ' %i job in the database %s'%(cnt, TABLE_NAME)

