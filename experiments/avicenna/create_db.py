from jobman import sql, DD, flatten, expand
from jobman.parse import filemerge

from experiments.avicenna.scripts import train_dA

def update_view(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    # user-friendly view
    db.createView(TABLE_NAME + 'view')

def first_xp(TABLE_NAME, batchsize=20):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    cnt =0
    state = DD()
    prefix = '/data/lisatmp/ift6266h11/' + TABLE_NAME + '/'

    state.dataset = 'avicenna'
    state.nhidden = 300
    state.tied_weights = True
    state.act_enc = 'sigmoid'
    state.act_dec = 'sigmoid'
    state.batchsize = batchsize
    state.epochs = 50
    state.cost_type = 'CE'
    state.noise_type = 'gaussian'
    state.normalize = False

    for unsuplr in [1e-2,1e-4]:
        for noise in [0.2, 0.4]:
            for nh in [100,150,250]:
                cnt += 1
                state.lr = unsuplr
                state.nhidden=nh
                state.corruption_level = noise
                state.savedir = prefix + str(cnt) + '/'
                sql.insert_job(train_dA, flatten(state), db)

    update_view(TABLE_NAME)
    print ' %i job in the database %s'%(cnt, TABLE_NAME)

