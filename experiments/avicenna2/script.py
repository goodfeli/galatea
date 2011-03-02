# Standard library imports
import os
import getpass
from itertools import izip

# Third-party imports
from jobman import sql, DD, flatten

# Local imports
from framework import utils
from framework.scripts.experiment import train_da, train_pca
from posttraitement.pca import PCA

def update_view(table_name):
    """ Create a user-friendly view """
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/' + table_name)
    db.createView(table_name + 'view')

def run_da(conf, channel):
    """ Main routine to train the DA after applying a PCA to the dataset """
    data = utils.load_data(conf)
    
    # Load the PCA
    pca = PCA.load(conf['pca_dir'], 'model-pca.pkl')
    pca_fn = pca.function('pca_transform_fn')
    
    data_after_pca = [utils.sharedX(pca_fn(set.get_value()), borrow=True)
                      for set in data]
    del data
    
    # Train a DA over the computed representation
    train_da(conf, data_after_pca)
    del data_after_pca
    
    return channel.COMPLETE


def run_pca(conf):
    """ Main routine to train a simple PCA """
    data = utils.load_data(conf)
    
    # Train the PCA
    data_blended = utils.blend(conf, data)
    train_pca(conf, data_blended)
    del data_blended
    

def create(table_name):
    conf = {# DA arguments
            'n_hid': 200,
            'lr_anneal_start': 100,
            'base_lr': 1e-3,
            'tied_weights': True,
            'act_enc': 'sigmoid',
            'act_dec': None,
            'irange': 0.001,
            'cost_class' : 'MeanSquaredError',
            'corruption_class' : '',
            'corruption_level': 0.,
            # PCA arguments
            'num_components': 75,
            'min_variance': 0,
            'whiten': True,
            # Experiment arguments
            'dataset' : 'avicenna',
            'expname' : 'pca75dae20ep',
            'batch_size' : 20,
            'epochs' : 20,
            'train_prop' : 1,
            'valid_prop' : 2,
            'test_prop' : 2,
            'normalize' : True,
            'normalize_on_the_fly' : False,
            'randomize_valid' : True,
            'randomize_test' : True,
            'saving_rate': 2,
            'alc_rate' : 2,
            'resulting_alc' : True,
            'pca_dir' : '',
            }
    conf = DD(conf)

    # Job-dependant parameters
    noise_type = ['BinomialCorruptor', 'GaussianCorruptor']
    noise_level = [[.3, .5], [.1, .3]]
    var_nhidden = [50, 200, 500, 1000]
    
    # Database creation
    db = sql.db('postgres://ift6266h11@gershwin.iro.umontreal.ca/ift6266h11_sandbox_db/' + table_name)
    base_dir = os.path.join('/data/lisatmp/ift6266h11/', getpass.getuser())
    conf['pca_dir'] = os.path.join(base_dir, 'pca')
    base_dir = os.path.join(base_dir, table_name)
	# If you don't have a 'model-pca.pkl' in your 'pca_dir' folder, remember to train it
    #run_pca(conf)
    counter = 0
    
    for nh in var_nhidden:
        for nt, level in izip(noise_type, noise_level):
            for nl in level:
                conf['n_hid'] = nh
                conf['corruption_class'] = nt
                conf['corruption_level'] = nl
                
                # Insert into db
                s = db.session()
                try:
                    inserted = sql.insert_job(run_da, flatten(conf), db, session=s)
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
