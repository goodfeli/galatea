"""kmeans test with the new framework."""
# Standard library imports
import time
import sys
import os

# Third-party imports
import numpy
import theano
from theano import tensor
import cPickle
# Local imports
try:
    import framework
except ImportError:
    print >> sys.stderr, \
            "Framework couldn't be imported. Make sure you have the " \
            "repository root on your PYTHONPATH (or as your current " \
            "working directory)"
    sys.exit(1)

from auc import embed
from framework import utils
from framework.kmeans import KMeans
from pca_lcn import store_PCA_rita

import zipfile
from tempfile import TemporaryFile

#--------------------------------------------------------
def create_pca_lcn(conf,nb_comp,model):
    store_PCA_rita(nb_comp,conf['data_path'],model)
    return None

#--------------------------------------------------------	
def train_kmeans(conf):#,train,valid,test):
        
    kmeans = KMeans(conf['k_means'])
    
    #Load the previous dataset.
    train = cPickle.load(open(conf['data_path'] + 'train_postPCA_'+ str(conf['nb_comp'])+ 'comp.pkl'))
    valid = cPickle.load(open(conf['data_path'] + 'valid_postPCA_'+str(conf['nb_comp'])+'comp.pkl'))
    test = cPickle.load(open(conf['data_path'] + 'test_postPCA_'+str(conf['nb_comp'])+'comp.pkl'))
    
    kmeans.train(train)
    kmeans.save(conf['savedir']+'KMeans'+str(conf['k_means'])+'.pkl')
    print '==Debut de la valid=='
    output_valid = kmeans(valid)
    print '===Debut Du Test=='
    output_test = kmeans(test)
    print '==Sauvegarde des output=='
    filename = os.path.join(conf['savedir'], 'KMeans'+ str(conf['k_means'])+'_'+str(conf['nb_comp']) + 'comp.pkl')
    kmeans.save(filename)
    #utils.create_submission(conf,output_valid,output_test)

#--------------------------------------------------------	
def load_lcn_data(conf):
       
    #Load the previous dataset.
    train = cPickle.load(open(conf['data_path'] + 'train_postPCA_'+ str(conf['nb_comp'])+ 'comp.pkl'))
    valid = cPickle.load(open(conf['data_path'] + 'valid_postPCA_'+str(conf['nb_comp'])+'comp.pkl'))
    test = cPickle.load(open(conf['data_path'] + 'test_postPCA_'+str(conf['nb_comp'])+'comp.pkl'))
    return train,valid,test

#--------------------------------------------------------		
def submit(valid, test, filepath, valid_fn="harry_lisa_valid.prepro", test_fn="harry_lisa_final.prepro"):
    valid_file = TemporaryFile()
    test_file = TemporaryFile()
            
    numpy.savetxt(valid_file, valid, fmt="%.3f")
    numpy.savetxt(test_file, test, fmt="%.3f")
                        
    valid_file.seek(0)
    test_file.seek(0)
                                  
    submission = zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED)
                                            
    submission.writestr(valid_fn, valid_file.read())
    submission.writestr(test_fn, test_file.read())
                                                      
    submission.close()
    valid_file.close()
    test_file.close()

#--------------------------------------------------------	
if __name__ == '__main__':

    conf = { 
          'dataset' : 'harry',
          'expname' : 'hkl', # Used to create the submission file
          'transfer' : False,
          'normalize' : True, # (Default = True)
          'normalize_on_the_fly' : False, # (Default = False)
          'randomize_valid' : False, # (Default = True)
          'randomize_test' : False, # (Default = True)
          'saving_rate': 2, # (Default = 0)
          'savedir' : '/data/lisatmp/ift6266h11/harry_kmeans/outputs/',
          'data_path' : '/data/lisa/exp/dauphiya/harry/harry_train_1585.npy',
          'nb_comp' : 1585,        
          'k_means' : 3 ,
    }

    #create_pca_lcn(conf,conf['nb_comp'],model=filename)
    #Load the previous dataset.
    #train = cPickle.load(open(conf['data_path']))
	
    try:	
        train = numpy.load('/data/lisa/exp/dauphiya/harry/harry_train_1585.npy')
        #cPickle.dump(i, open('/u/bourretl/repos/temp/ift6266h11/experiments/harry2/harry_train_1585.pkl', 'wb'))
        #train = cPickle.load(open('/u/bourretl/repos/temp/ift6266h11/experiments/harry2/harry_train_1585.pkl'))
	
        valid = numpy.load('/data/lisa/exp/dauphiya/harry/harry_valid_1585.npy')
        #cPickle.dump(i, open('/u/bourretl/repos/temp/ift6266h11/experiments/harry2/harry_valid_1585.pkl', 'wb'))
        #valid = cPickle.load(open('/u/bourretl/repos/temp/ift6266h11/experiments/harry2/harry_valid_1585.pkl'))
	
        test = numpy.load('/data/lisa/exp/dauphiya/harry/harry_test_1585.npy')
        #cPickle.dump(i, open('/u/bourretl/repos/temp/ift6266h11/experiments/harry2/harry_test_1585.pkl', 'wb'))
        #test = cPickle.load(open('/u/bourretl/repos/temp/ift6266h11/experiments/harry2/harry_test_1585.pkl'))
    except:
        print "Unexpected error:", sys.exc_info()[0], sys.exc_info()[1]
        raise
	
    print ('dump completed')
	
    kmeans = KMeans(conf['k_means'])
		
    print ('==Debut du Train=='+'KMeans' + str(conf['k_means']) + '_' + str(conf['nb_comp']) + 'comp')
		
    kmeans.train(train)
    kmeans.save(conf['savedir']+'KMeans' + str(conf['k_means'])+'.pkl')
    print ('==Debut de la valid==' +'KMeans' + str(conf['k_means']) + '_' + str(conf['nb_comp']) + 'comp')
    output_valid = kmeans(valid)
    print ('===Debut Du Test==' +'KMeans' + str(conf['k_means']) + '_' + str(conf['nb_comp']) + 'comp')
    output_test = kmeans(test)
    print ('==Sauvegarde des output==' +'KMeans' + str(conf['k_means']) + '_' + str(conf['nb_comp']) + 'comp')
    filename = os.path.join(conf['savedir'], 'KMeans'+ str(conf['k_means']) + '_' + str(conf['nb_comp']) + 'comp')
    kmeans.save(filename + '.pkl')
    
    print '==submission created'
    submit(output_valid, output_test,  filename + '_SUBMISSION_.zip')
