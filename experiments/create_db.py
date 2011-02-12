from jobman import sql, DD, flatten, expand
from jobman.parse import filemerge

from experiments.scripts import train_dA

def update_view(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    # user-friendly view
    db.createView(TABLE_NAME + 'view')

def first_xp(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/'+TABLE_NAME)
    cnt = 0
    state = DD()
    prefix = '/data/lisa/exp/mesnilgr/ift6266h11/' + TABLE_NAME + '/'

    state.dataset = 'ule'
    state.nhidden = 500
    state.tied_weights = True
    state.act_enc = 'sigmoid'
    state.act_dec = 'sigmoid'
    state.batchsize = 1
    state.epochs = 50
    state.cost_type = 'CE'
    state.noise_type = 'gaussian'

    for unsuplr in [1e-2, 5e-2, 1e-3, 5e-4, 12]:
        for noise in [0.1, 0.3, 0.5]:
            state.lr = unsuplr
            state.corruption_level = noise

            # Insert into db
            s = db.session()
            try:
                inserted = sql.insert_job(train_dA, flatten(state), db, session=s)
                if inserted is not None:
                    # Count number of inserted jobs
                    cnt += 1
                    # Get job id to compute savedir
                    job_id = inserted.id
                    savedir = prefix + str(job_id) + '/'
                    # Insert savedir into the data base
                    inserted._set_in_session('savedir', savedir, s)
                    s.commit()
            finally:
                s.close()

    update_view(TABLE_NAME)
    print '  Inserted %i jobs the database %s'%(cnt, TABLE_NAME)

