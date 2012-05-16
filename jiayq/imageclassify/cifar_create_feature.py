import jiayq.imageclassify.pipeline as pipeline
import jiayq.imageclassify.datasets as datasets
from jiayq.utils import mpiutils
import sys
import cPickle
import os
import argparse
from scipy import io

mpiutils.report()

parser = argparse.ArgumentParser(description="Script to generate cifar features.", \
                                 epilog="Yangqing Jia at Berkeley, 2011")
parser.add_argument('--mirror',action='store_true',help='whether to mirror training image')
parser.add_argument('-d','--datadir',type=str,default=None,help='data directory')
parser.add_argument('--dataname',type=str,default='cifar',help='dataset name')
parser.add_argument('-r','--rootfolder',default='./',help='the root folder')
parser.add_argument('-s', '--patchsize',type=int,default=6,help='patch size')
parser.add_argument('-g','--isGrey',action='store_true',help='whether to use color image')
parser.add_argument('-l','--normalize',type=str,default='meanvar',help='patch normalization method')
parser.add_argument('-o','--preprocess',type=str,action='append',help='preprocess method')
parser.add_argument('-n','--dictSize',type=int,default=1600,help='the dictionary size')
parser.add_argument('-m','--dictMethod',type=str,default='omp',help='the dictionary training method')
parser.add_argument('-e','--encoding', type=str, default='thres',help='the encoding method')
parser.add_argument('-b','--bins', type=int,default=4,help='the number of spatial bins per dimension')
parser.add_argument('-p','--pooling', type=str,default='max',help='the pooling method')
parser.add_argument('--extractor', type=str, default=None, help='read from a previously pickled extractor')
parser.add_argument('--preprocessor', type=str, default=None, help='read from a previously pickled preprocessor')
parser.add_argument('--encoder', type=str, default=None, help='read from a previously pickled encoder')
parser.add_argument('--pooler', type=str, default=None, help='read from a previously pickled pooler')
mpiutils.rootprint(str(sys.argv))
args = parser.parse_args(sys.argv[1:])

args.isColor = (not args.isGrey)

subdir = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                       args.dataname,\
                                       args.mirror,\
                                       args.patchsize,\
                                       args.isColor,\
                                       args.normalize,\
                                       '_'.join(args.preprocess),\
                                       args.dictSize,\
                                       args.dictMethod,\
                                       args.encoding,\
                                       args.bins,\
                                       args.pooling)
mpiutils.rootprint('save to subdir {}'.format(subdir))
outputfolder = os.path.join(args.rootfolder, subdir)

if mpiutils.rank == 0:
    if not os.path.exists(outputfolder):
        print 'Creating folder {}'.format(outputfolder)
        # the try-catch below solves race condition
        try:
            os.makedirs(outputfolder)
        except OSError:
            pass
    print 'loading data...'
    if args.dataname == 'cifar':
        if args.mirror:
            cifar = datasets.CifarDatasetMirror(args.datadir)
        else:
            cifar = datasets.CifarDataset(args.datadir)
    elif args.dataname == 'mnist':
        # TODO: This is the temporary fix during cvpr crunch.
        # clean the names after this rush.
        cifar = datasets.MNISTDataset(args.datadir)
    elif args.dataname[:5] == 'mnist':
        digits = [int(i) for i in list(args.dataname[5:])]
        print 'digits: ', digits
        cifar = datasets.MNISTDatasetSub(digits,args.datadir)
        print 'Ntrain: {}'.format(cifar.Ntrain)
        print 'Ntest: {}'.format(cifar.Ntest)
else:
    cifar = None
cifar = mpiutils.comm.bcast(cifar,root=0)

if args.encoder is not None and args.preprocessor is None:
    print 'Error: if you define an encoder, you should provide the corresponding preprocessor'
    
if args.extractor is not None:
    mpiutils.rootprint('loading extractor')
    f = open(args.extractor,'r')
    unpickler = cPickle.Unpickler(f)
    extractor = unpickler.load()
else:
    nChannels = 3 if args.isColor else 1
    extractor = pipeline.PatchExtractor(args.patchsize,nChannels,normalize=args.normalize)
if args.preprocessor is not None:
    mpiutils.rootprint('loading preprocessor')
    f = open(args.preprocessor,'r')
    unpickler = cPickle.Unpickler(f)
    preprocessor = unpickler.load()
else:
    preprocessor = None
    for i in range(len(args.preprocess)):
        preprocessor = pipeline.PatchPreprocessor(args.preprocess[i],previousPrep = preprocessor)
if args.encoder is not None:
    mpiutils.rootprint('loading encoder')
    f = open(args.encoder,'r')
    unpickler = cPickle.Unpickler(f)
    encoder = unpickler.load()
else:
    encoder = pipeline.PatchEncoder(args.encoding)
if args.pooler is not None:
    f = open(args.pooler,'r')
    unpickler = cPickle.Unpickler(f)
    pooler = unpickler.load()
else:
    pooler = pipeline.SpatialPooler(args.bins,args.pooling)

# even if we do not use dictTrainer, we initialize one
dictTrainer = pipeline.DictTrainer(args.dictSize, method=args.dictMethod, specs={'maxiter':100})

pipeliner = pipeline.Pipeliner(extractor,preprocessor,dictTrainer,encoder,pooler)

if args.encoder is None:
    mpiutils.rootprint('training...')
    pipeliner.train(cifar,400000)
mpiutils.rootprint('Processing data...')
# save the labels and the pipeliner first
if mpiutils.rank == 0:
    print 'tr label size: {}'.format(cifar.label_tr.shape)
    print 'te label size: {}'.format(cifar.label_te.shape)
    io.savemat(outputfolder+'/tr_label.mat',{'label':cifar.label_tr}, oned_as='row')
    io.savemat(outputfolder+'/te_label.mat',{'label':cifar.label_te}, oned_as='row')
    cPickle.Pickler(open(outputfolder+'/extractor.dat','w')).dump(extractor)
    cPickle.Pickler(open(outputfolder+'/preprocessor.dat','w')).dump(preprocessor)
    cPickle.Pickler(open(outputfolder+'/dictTrainer.dat','w')).dump(dictTrainer)
    cPickle.Pickler(open(outputfolder+'/encoder.dat','w')).dump(encoder)
    cPickle.Pickler(open(outputfolder+'/pooler.dat','w')).dump(pooler)
    
# process the data
pipeliner.batch_process_dataset(cifar, 1000, outputfolder+'/cifar_tr_{}_{}.mat')
pipeliner.batch_process_dataset(cifar, 1000, outputfolder+'/cifar_te_{}_{}.mat',fromTraining=False)
mpiutils.safebarrier()
mpiutils.rootprint('Done.')
