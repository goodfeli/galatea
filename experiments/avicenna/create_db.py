from jobman import sql, DD, flatten, expand
from jobman.parse import filemerge

from experiments.avicenna.scripts import train_dA,train_SDA

def update_view(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    # user-friendly view
    db.createView(TABLE_NAME + 'view')

def first_xp(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    cnt =0
    state = DD()
    prefix = '/data/lisatmp/ift6266h11/bissonva/' + TABLE_NAME + '/'

    #default params are best ones so far
    state.dataset = 'avicenna'
    state.nhidden = 1000
    state.tied_weights = True
    state.act_enc = 'tanh'
    state.act_dec = 'linear'
    state.batch_size = 20
    state.epochs = 100
    state.cost_type = 'MSE'
    state.noise_type = 'gaussian'
    state.normalize = True
    state.lr = 1e-3
    state.corruption_level=.2

    testing=False
    if testing:
        for nhidden in [1500]:
            for batch_size in [15,35]:
                for epochs in [100,150]:
                    for corruption_level in [.1,.2]:
                        state.nhidden=nhidden
                        state.epochs=epochs
                        state.corruption_level=corruption_level
                        state.batch_size=batch_size
                        cnt += 1
                        state.savedir = prefix + str(cnt) + '/'
                        sql.insert_job(train_dA, flatten(state), db)
    else:
        sql.insert_job(train_dA, flatten(state), db)
        cnt=1

    update_view(TABLE_NAME)
    print ' %i job in the database %s'%(cnt, TABLE_NAME)

def sda_xp(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    cnt =0
    state = DD()
    prefix = '/data/lisatmp/ift6266h11/bissonva/' + TABLE_NAME + '/'

    state.dataset = 'avicenna'
    state.nhidden = 500
    state.tied_weights = True
    state.act_enc = 'tanh'
    state.act_dec = 'linear'
    state.batch_size = 20
    state.pretraining_epochs= 150
    state.cost_type = 'MSE'
    state.noise_type = 'gaussian'
    state.normalize = False
    state.pretrain_lr = 1e-3

    state.finetune_lr = 0.1
    state.training_epochs = 100

    for training_epochs in [50,100,150]:
        for finetune_lr in [1e-2,1e-3,1e-4]:
                state.training_epochs=training_epochs
                state.finetune_lr=finetune_lr
                cnt += 1
                state.savedir = prefix + str(cnt) + '/'
                sql.insert_job(train_SDA, flatten(state), db)

    update_view(TABLE_NAME)
    print ' %i job in the database %s'%(cnt, TABLE_NAME)

