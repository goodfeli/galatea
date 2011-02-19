from jobman import sql, DD, flatten, expand
from jobman.parse import filemerge

from experiments.avicenna.scripts import train_dA

def update_view(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    # user-friendly view
    db.createView(TABLE_NAME + 'view')

def first_xp(TABLE_NAME):
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
    state.epochs = 150
    state.cost_type = 'MSE'
    state.noise_type = 'gaussian'
    state.normalize = False
    state.lr = 1e-23
    state.corruption_level=.2

    for nhidden in [750,500,1000]:
        for batch_size in [15,35]:
            for corruption_level in [.1,.2]:
                state.corruption_level=corruption_level
                state.batch_size=batch_size
                cnt += 1
                state.savedir = prefix + str(cnt) + '/'
                sql.insert_job(train_dA, flatten(state), db)

    update_view(TABLE_NAME)
    print ' %i job in the database %s'%(cnt, TABLE_NAME)

