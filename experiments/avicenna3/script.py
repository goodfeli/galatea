# Standard library imports
import os
import getpass
from itertools import izip

# Third-party imports
from jobman import sql, DD, flatten

# Local imports
from experiments.avicenna3.experiment import first_xp

def update_view(table_name):
    """ Create a user-friendly view """
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/' + table_name)
    db.createView(table_name + 'view')

def create(table_name): 

    # Job-dependant parameters
    var_lr = [0.1, 0.05, 0.01]
    
    # Database creation
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/' + table_name)
    base_dir = os.path.join('/data/lisatmp/ift6266h11/', getpass.getuser())
    base_dir = os.path.join(base_dir, table_name)
    counter = 0
    state = DD()

    for lr in var_lr:
        # Insert into db
        state.lr = lr
        s = db.session()
        try:
            inserted = sql.insert_job(first_xp, flatten(state), db, session=s)
            if inserted is not None:
                counter += 1
                # Get job id to compute savedir
                job_id = inserted.id
                save_dir = os.path.join(base_dir, str(job_id))
                # Insert save_dir into the data base
                inserted._set_in_session('da_dir', save_dir, s)
                s.commit()
        finally:
            s.close()

    update_view(table_name)
    print '... %i jobs inserted in the database %s' % (counter, table_name)
