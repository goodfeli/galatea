"""
	Functions related to the UTLC data set
"""
import os, gzip, cPickle

import numpy
import theano

import pylearn.datasets.config as config
import pylearn.io.filetensor as ft

# for functools.partial
import functools as fc

datasets = ["avicenna", "harry", "rita", "sylvester", "terry", "ule"]

datasets_params = {    #feat  #dev    #trans  #val  #fin
		"avicenna"  : [  120, 150205,  50000, 4096, 4096],
		"harry"     : [ 5000,  69652,  20000, 4096, 4096],
		"rita"      : [ 7200, 111808,  24000, 4096, 4096],
		"sylvester" : [  100, 572820, 100000, 4096, 4096],
		"terry"     : [47236, 217034,  40000, 4096, 4096],
		"ule"       : [  784,  26808,  10000, 4096, 4096] }

# normalize functions
def normalize_gaussian(mean, std, set):
	return (set.astype(theano.config.floatX) - mean) / std

def normalize_maximum(max, set):
	return set.astype(theano.config.floatX) / max

# statistics
avi_mean = 514.62248464301717
avi_std  = 6.901698313710936
harry_std = 1
rita_max = 290
syl_mean = 0
syl_std  = 1
terry_max = 300
ule_max = 255

datasets_normalizer= {
		"avicenna"  : [False, fc.partial(normalize_gaussian, avi_mean, avi_std)],
		"harry"     : [False, fc.partial(normalize_maximum, harry_std)],
		"rita"      : [False, fc.partial(normalize_maximum, rita_max)],
		"sylvester" : [False, fc.partial(normalize_gaussian, syl_mean, syl_std)],
		"terry"     : [ True, fc.partial(normalize_maximum, terry_max)],
		"ule"       : [False, fc.partial(normalize_maximum, ule_max)] }

class dataset(object):
	"""
	A class for loading an UTLC dataset into memory.
	"""

	def __init__(self, name, normalized=True, sparse=False):
		assert name in datasets
		# data information
		# TODO learn indentation tricks
		self.name = name
		self.n_feats, self.n_devel, self.n_trans, self.n_valid, self.n_final = datasets_params.get(name)

		# subsets
		self.devel_set  = self.valid_set  = self.final_set  = None
		self.devel_path = self.valid_path = self.final_path = None

		# data manipulation
		self.normalized = normalized
		self.sparse, self.normalize = datasets_normalizer.get(name)

		# data loader
		self.loader = ft.read
		if self.sparse:
			self.loader = cPickle.load
		self.load_dataset()

	def set_filepath(self):
		folder = 'filetensor'
		extension = '.ft'
		# root = config.data_root()
		root = "/data/lisa/data/"

		if self.sparse:
			folder = 'sparse'
			extension = '.npy'

		# TODO, fix mapping
		# devel = train
		# valid = valid
		# final = test
		self.devel_path = os.path.join(root,'UTLC', folder,
							self.name+'_'+'train'+extension)
		self.valid_path = os.path.join(root,'UTLC', folder,
							self.name+'_'+'valid'+extension)
		self.final_path = os.path.join(root,'UTLC', folder,
							self.name+'_'+'test'+extension)

	def load_dataset(self):
		self.set_filepath()

		self.devel_set = load_fromfile(self.devel_path, self.loader)
		self.final_set = load_fromfile(self.final_path, self.loader)
		self.valid_set = load_fromfile(self.valid_path, self.loader)

		if self.normalized:
			for set in ["devel_set", "final_set", "valid_set"]:
				self.__dict__[set] = self.normalize(self.__dict__[set])
				# in place modification saves a lot of space
				#dataset = self.__dict__[set]
				#for i in range(dataset.shape[0]):
				#	for j in range(self.n_feats):
				#		value = dataset[i][j]
				#		dataset[i][j] = self.normalize(value)

# Utiliy to load our file
def load_fromfile(fname, loader):
    f = None
    try:
        if not os.path.exists(fname):
            fname = fname+'.gz'
            assert os.path.exists(fname)
            f = gzip.open(fname)
        else:
            f = open(fname)
        d = loader(f)
    finally:
        if f:
            f.close()

    return d
