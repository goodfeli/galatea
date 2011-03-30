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
          'dataset' : 'rita_',
          'expname' : 'rkl', # Used to create the submission file
          'transfer' : False,
          'normalize' : True, # (Default = True)
          'normalize_on_the_fly' : False, # (Default = False)
          'randomize_valid' : False, # (Default = True)
          'randomize_test' : False, # (Default = True)
          'saving_rate': 2, # (Default = 0)
          'savedir' : '/data/lisatmp/ift6266h11/rita_kmeans_pca_lcn/outputs/',
          'data_path' : '/data/lisatmp/ift6266h11/rita_kmeans_pca_lcn/',
          'nb_comp' : 1000,        
          'k_means' : 10,
    }
	
    filename = model=conf['data_path']+'train_postPCA_'+str(conf['nb_comp'])+'comp.pkl'
    
    componentsNum = [1000, 750, 100]
    kmeansNum = [200, 175, 100] 	
	
    for it in range(3):
        i = componentsNum[it]
        j = kmeansNum[it]
		
        create_pca_lcn(conf,i,model=filename)
        #Load the previous dataset.
        train = cPickle.load(open(conf['data_path'] + 'train_postPCA_'+ str(i)+ 'comp.pkl'))
        print '==Load Train KMeans' + str(j) + '_' + str(i) + 'comp =='
        valid = cPickle.load(open(conf['data_path'] + 'valid_postPCA_'+str(i)+'comp.pkl'))
        print '==Load Valid KMeans' + str(j) + '_' + str(i) + 'comp =='
        test = cPickle.load(open(conf['data_path'] + 'test_postPCA_'+str(i)+'comp.pkl'))
        print '==Load Test KMeans' + str(j) + '_' + str(i) + 'comp =='

        kmeans = KMeans(j)
		
        print '==Debut du Train KMeans' + str(j) + '_' + str(i) + 'comp =='
		
        kmeans.train(train)
        kmeans.save(conf['savedir']+'KMeans' + str(j)+'.pkl')
        print ('==Debut de la valid==' +'KMeans' + str(j) + '_' + str(i) + 'comp')
        output_valid = kmeans(valid)
        print ('===Debut Du Test==' +'KMeans' + str(j) + '_' + str(i) + 'comp')
        output_test = kmeans(test)
        print ('==Sauvegarde des output==' +'KMeans' + str(j) + '_' + str(i) + 'comp')
        filename = os.path.join(conf['savedir'], 'KMeans'+ str(j) + '_' + str(i) + 'comp')
        kmeans.save(filename + '.pkl')
    
    submit(output_valid, output_test,  filename + '_SUBMISSION_.zip')
