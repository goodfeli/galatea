from jobman import sql, DD, flatten, expand
from jobman.parse import filemerge

from experiments.sylvester.scripts import train_dA

def update_view(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    # user-friendly view
    db.createView(TABLE_NAME + 'view')

def first_xp(TABLE_NAME, batchsize):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    cnt =0
    state = DD()
    prefix = '/data/lisatmp/ift6266h11/' + TABLE_NAME + '/'

    state.dataset = 'sylvester'
    state.nhidden = 500
    state.tied_weights = True
    state.act_enc = 'sigmoid'
    state.act_dec = 'sigmoid'
    state.batchsize = batchsize
    state.epochs = 50
    state.cost_type = 'CE'
    state.noise_type = 'gaussian'
    state.normalize = False

    for unsuplr in [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]:
        for noise in [0.1, 0.3, 0.5]:
            cnt += 1
            state.lr = unsuplr
            state.corruption_level = noise
            state.savedir = prefix + str(cnt) + '/'
            sql.insert_job(train_dA, flatten(state), db)

    update_view(TABLE_NAME)
    print ' %i job in the database %s'%(cnt, TABLE_NAME)

