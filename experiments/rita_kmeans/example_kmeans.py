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
def create_pca_lcn(conf,nb_comp,model):
    store_PCA_rita(nb_comp,conf['data_path'],model)
    return None
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
    utils.create_submission(conf,output_valid,output_test)
def load_lcn_data(conf):
       
    #Load the previous dataset.
    train = cPickle.load(open(conf['data_path'] + 'train_postPCA_'+ str(conf['nb_comp'])+ 'comp.pkl'))
    valid = cPickle.load(open(conf['data_path'] + 'valid_postPCA_'+str(conf['nb_comp'])+'comp.pkl'))
    test = cPickle.load(open(conf['data_path'] + 'test_postPCA_'+str(conf['nb_comp'])+'comp.pkl'))
    return train,valid,test

if __name__ == '__main__':

    conf = { 
          'dataset' : 'rita_',
          'expname' : 'dummy', # Used to create the submission file
          'transfer' : False,
          'normalize' : True, # (Default = True)
          'normalize_on_the_fly' : False, # (Default = False)
          'randomize_valid' : False, # (Default = True)
          'randomize_test' : False, # (Default = True)
          'saving_rate': 2, # (Default = 0)
          'savedir' : '/data/lisatmp/ift6266h11/rita-pca-lcn/outputs/',
          'data_path' : '/data/lisatmp/ift6266h11/rita-pca-lcn/',
          'nb_comp' : 1000,        
          'k_means' : 10 ,
    }
                
    create_pca_lcn(conf,conf['nb_comp'],model=conf['data_path']+'train_postPCA_'+str(conf['nb_comp'])+'comp.pkl')
    #Load the previous dataset.
    train = cPickle.load(open(conf['data_path'] + 'train_postPCA_'+ str(conf['nb_comp'])+ 'comp.pkl'))
    valid = cPickle.load(open(conf['data_path'] + 'valid_postPCA_'+str(conf['nb_comp'])+'comp.pkl'))
    test = cPickle.load(open(conf['data_path'] + 'test_postPCA_'+str(conf['nb_comp'])+'comp.pkl'))
    

    kmeans = KMeans(conf['k_means'])
    
    print '==Debut du Train=='
    
    kmeans.train(train)
    kmeans.save(conf['savedir']+'KMeans'+str(conf['k_means'])+'.pkl')
    print '==Debut de la valid=='
    output_valid = kmeans(valid)
    print '===Debut Du Test=='
    output_test = kmeans(test)
    print '==Sauvegarde des output=='
    filename = os.path.join(conf['savedir'], 'KMeans'+ str(conf['k_means'])+'_'+str(conf['nb_comp']) + 'comp.pkl')
    kmeans.save(filename)
    
    utils.create_submission(conf,output_valid,output_test)
