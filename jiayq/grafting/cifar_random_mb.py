import grafting_mb
import sys
import numpy as np
import scipy as sp
import os
import argparse
from utils import mpi
from utils.timer import Timer
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

parser = argparse.ArgumentParser(description="Script to test cifar with existing trained dump.", \
                                 epilog="Yangqing Jia at NECLA, 2011")
parser.add_argument('-r', '--data_root', default='.', help='the dataset path')
parser.add_argument('-n', '--nBinsPerEdge', type=int, default=0, help='the number of bins per edge')
parser.add_argument('-d', '--nCodes', type=int, default=0, help='the number of codes')
parser.add_argument('-b', '--batch_size', type=int, default=1000, help='the batch size that the data is stored')
parser.add_argument('-m', '--maxSelFeat', type=int, default=6400, help='max number of selected features')
parser.add_argument('-g', '--gamma', type=float, default=0.01, help='regularization term for classification')
parser.add_argument('-e', '--local_cache_root', default=None, help='local cache root')
parser.add_argument('-l', '--read_local_cache', type=int, default=0, help='whether to read local cache or not')
parser.add_argument('-c', '--nClass', type=int, default=10, help='number of classes')
parser.add_argument('-t', '--random_iterations', type=int, default=1, help='number of random iterations')
parser.add_argument('-s', '--skip_normalization',  default=False, action = 'store_true')
mpi.rootprint(str(sys.argv))
args = parser.parse_args(sys.argv[1:])

# cifar specifications
data_file = 'cifar_tr_{}_{}.mat'
label_file = 'tr_label.mat'
test_data_file = 'cifar_te_{}_{}.mat'
test_label_file = 'te_label.mat'
nTraining = 50000
nTesting = 10000

grafter = grafting_mb.GrafterMPI()
tester = grafting_mb.GrafterMPI()

grafter.init_specs(nTraining, args.nBinsPerEdge, args.nCodes, args.nClass, args.maxSelFeat, args.gamma, np.float64)
tester.init_specs(nTesting, args.nBinsPerEdge, args.nCodes, args.nClass, args.maxSelFeat, args.gamma, np.float64)

if args.local_cache_root:
    args.local_cache_root = os.path.join(args.local_cache_root, str(rank))

grafter.load_data_batch(args.data_root, args.batch_size, data_file, label_file,\
                        rootRead = True, \
                        local_cache_root = args.local_cache_root, read_local_cache = args.read_local_cache,
                        should_normalize = not args.skip_normalization)
tester.load_data_batch(args.data_root, args.batch_size, test_data_file, test_label_file,\
                       rootRead = True, isTest=True, \
                       local_cache_root = args.local_cache_root, read_local_cache = args.read_local_cache)

grafter.randomselecttest(tester, args.random_iterations, should_normalize = not args.skip_normalization)

