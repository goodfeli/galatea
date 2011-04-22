import numpy

from jobman import sql, DD, flatten, expand
from jobman.parse import filemerge

from .terry_ssrbm import experiment0

TABLE_NAME='terry_ssrbm_'

def update_view(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_db/'+TABLE_NAME)
    # user-friendly view
    db.createView(TABLE_NAME + 'view')

def zarathustra(TABLE_NAME):
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_db/'+TABLE_NAME)
    cnt = 0
    state = DD()
    prefix = '/data/lisa/exp/mesnilgr/ift6266h11/' + TABLE_NAME + '/'

    # Default state is defined in the experiment function
    layer1 = DD()
    state.layer1 = layer1

    meta_seed = 42
    rng = numpy.random.RandomState(seed=42)
    hp_list = []

    for nhid in (50, 100, 200, 500):
        for alpha0 in (4, 5, 7, 10):
            for n_s_per_h in (1, 2):
                for B0 in (3, 10, 30):
                    for base_lr in (1e-1, 3e-2, 1e-2, 3e-3):
                        seed = rng.randint(2**30)
                        hp_list.append((seed, nhid, alpha0, n_s_per_h, B0, base_lr))

    n_hp = len(hp_list)
    n_exp = 100
    perm = rng.permutation(n_hp)

    exp_list = [hp_list[i] for i in perm][:n_exp]


    for seed, nhid, alpha0, n_s_per_h, B0, base_lr in exp_list:
        layer1.seed = seed
        layer1.nhid = nhid
        layer1.alpha0 = alpha0
        layer1.n_s_per_h = n_s_per_h
        layer1.B0 = B0
        layer1.base_lr = base_lr

        # Insert into db
        s = db.session()
        try:
            inserted = sql.insert_job(experiment0, flatten(state), db, session=s)
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

