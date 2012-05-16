'''
This code implements the sketch of the grafting algorithm described in

Grafting: fast, incremental feature selection by gradient descent in function space
S Perkins, K Lacker, J Theiler, JMLR 2003

Different from grafting.py, this is the memory-bound version, which only deal with
overcomplete bins based on a K*K grid. It runs slower than grafting.py - instead
of pre-computing and storing all the features, it computes features on the fly, and
only stores the base features for each bin of the K*K grid. Specifically, for
overcomplete bins on a K*K grid, with D dictionary entries and N data, grafting.py
needs K^2(K+1)^2DN/4*size(double) bytes to store the features, while this code
only needs K^2DN*size(double)+K^2*(K+1)^2/4*sizeof(bool) bytes. Note that K is
usually small, so the second term does not introduce too much storage overhead.
'''

__author__ = 'Yangqing Jia'
__status__ = 'Development'
__email__  = 'jiayq [at] eecs [dot] berkeley [dot] edu'

# system modules
import mpi4py.MPI as MPI
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.optimize as optimize
import os
import time
import warnings
# Yangqing's modules
from utils import mpi
from utils.timer import Timer
# local modules
import fastmax as fm
import loss as ls
import binsdef as bd

def dot_Asafe(A, B, out, matmul_batch=1024):
    '''
    do a save np.dot in case the size of A is too large, and in case
    we are using a buggy ATLAS that may cause int overflows when indexing
    a large matrix (which is the case for skyservers. I did not observer this
    on the Berkeley cluster, who uses numpy + Intel MKL, but maybe it's better
    to be cautious too). Note that A, B and out should all be matrices that
    are C contiguous.
    '''
    n_mm_batch = int(np.ceil(A.shape[0] / float(matmul_batch)))
    for bid in range(n_mm_batch):
        start = bid*matmul_batch
        end = min((bid+1)*matmul_batch, A.shape[0])
        # in newer numpy versions, we have np.dot(.,.,out=.)
        # but we will keep backward compatibility here.
        out[start:end] = np.dot(A[start:end], B)
    return

# Set True to use the C version of the loss functions.
if True:
    gL_bnll = ls.gL_bnll_c
    LgL_bnll = ls.LgL_bnll_c
else:
    EXP_MAX = np.float64(100.0) # exp(x) for any x value larger than this will return exp(EXP_MAX)

    def exp_safe(x):
        '''
        compute the safe exp 
        '''
        return np.exp(np.minimum(x,EXP_MAX))
        
    def gL_bnll(y,f):
        '''
        The BNLL gradient
        '''
        expnyf = exp_safe(-y*f+1)
        return -y*expnyf / (1.0+expnyf)
        
    def LgL_bnll(y,f):
        '''
        jointly computing the loss and gradient is usually faster
        '''
        expnyf = exp_safe(-y*f+1)
        return np.log(expnyf+1.0), -y*expnyf / (expnyf+1.0)

def ObjFuncIncrement(wb, X, y, currwxb, gamma):
    w = wb[:-1]
    b = wb[-1]
    f = np.dot(w,X) + currwxb + b
    L, gL = LgL_bnll(y,f)
    gwb = np.empty(wb.shape)
    gwb[:-1] = np.dot(X, gL) / X.shape[1] + gamma*w
    gwb[-1] = np.mean(gL)
    return np.mean(L) + gamma/2.0*np.sum(w**2), gwb
    
class GrafterMPI:
    '''
    The main grafter class, implemented with MPI support
    '''
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
    
    def safebarrier(self, tag=0, sleep=0.01):
        '''
        This is a better mpi barrier than MPI.comm.barrier(): the original barrier
        may cause idle processes to still occupy the CPU, while this barrier waits.
        '''
        comm = self.comm
        size = comm.Get_size() 
        if size == 1: 
            return 
        rank = comm.Get_rank() 
        mask = 1 
        while mask < size: 
            dst = (rank + mask) % size 
            src = (rank - mask + size) % size 
            req = comm.isend(None, dst, tag) 
            while not comm.Iprobe(src, tag): 
                time.sleep(sleep) 
            comm.recv(None, src, tag) 
            req.Wait() 
            mask <<= 1


    def init_specs(self, nData, nBinsPerEdge, nCodes, nLabel, maxGraftDim, gamma, dtype, \
                   metabinGenerator = bd.rectangularBins):
        '''
        Initialize the specs. Specifically, the raw data (for the base bins) is
        a nBinsPerEdge^2 * nCodes * nData cube, and each node will host a subset
        of the codes (all bins for any single code will be hosted on the same node).
        ==Parameters==
        nData: number of data points.
        nBinsPerEdge: number of base bins per edge. For example, for 4x4 base bins,
            pass 4.
        nCodes: the codebook size.
        nLabel: number of labels.
        maxGraftDim: the maximum number of features to select.
        gamma: regularizer for the classifier.
        dtype: data type. Only np.float64 is supported now as we have some c-code
            that has double-precision version only.
        metabinGenerator: the function to generate metabins. See bindef.py
        '''
        # determine feature range and data range
        if nData < self.size or nCodes < self.size:
            print 'Seriously? Is the problem really large scale?'
            # I know it's unethical, but whatever
            exit()
        self.nData = nData
        self.nCodes = nCodes
        self.nBinsPerEdge = nBinsPerEdge
        self.nBins = nBinsPerEdge*nBinsPerEdge
        self.nBaseFeat = self.nCodes*self.nBins
        self.metabins = metabinGenerator(nBinsPerEdge)
        self.nMetabins = self.metabins.shape[0]
        self.nLabel = nLabel
        if maxGraftDim > self.nMetabins*self.nCodes:
            mpi.rootprint('Warning: maxGraftDim should be no more than the number of available features.')
            maxGraftDim = self.nMetabins*self.nCodes
        self.maxGraftDim = maxGraftDim
        self.gamma = gamma
        self.dtype = dtype
        self.ncode_per_node = int(np.ceil( float(nCodes) / self.size ))
        self.codeRange = [self.ncode_per_node*self.rank, min(self.ncode_per_node*(self.rank+1), nCodes)]
        self.nCodeLocal = int(self.codeRange[1] - self.codeRange[0])
        self.mLocal = np.zeros((self.nCodeLocal, self.nMetabins), dtype=self.dtype)
        self.stdLocal = np.zeros((self.nCodeLocal, self.nMetabins), dtype=self.dtype)
        self.normalized = False
        # pre-allocate data space
        self.featSlice = np.zeros([self.nCodeLocal, self.nBins, self.nData], dtype=self.dtype)
        self.labels = -np.ones([self.nLabel, self.nData], dtype=self.dtype)
        self.rawlabels = np.zeros(self.nData, dtype=np.int)
        # pre-allocate selected features cache
        if self.rank < self.nLabel:
            self.dataSel = np.zeros([self.maxGraftDim, self.nData], dtype=self.dtype) # selected features
        else:
            self.dataSel = None
        # pre-allocate classifier parameters
        self.weights = np.zeros([self.nLabel, self.maxGraftDim], dtype=self.dtype) # weights
        self.b = np.zeros(self.nLabel, dtype=self.dtype) # bias
        self.curr_wxb = np.zeros([self.nLabel,self.nData], dtype=self.dtype) # current prediction
        # pre-allocate feature selection statistics
        self.nSelFeats = 0 # number of selected features
        self.selCodeID = np.zeros(self.maxGraftDim, dtype=np.int)
        self.selMetabinID = np.zeros(self.maxGraftDim, dtype=np.int)
        self.isSelected = np.zeros((self.nCodes, self.nMetabins),dtype=np.bool) # 0-1 array to define if a feature is selected
        # pre-allocate mpi buffer here
        self.featBuffer = np.zeros(self.nData, dtype = self.dtype)
        self.featBufferPerCode = np.zeros((self.nMetabins, self.nData), dtype=self.dtype)
        # other buffers
        self.localGradMat = np.zeros((self.nCodeLocal, self.nMetabins, self.nLabel), dtype = self.dtype)
        self.scoreVec = np.zeros((self.nCodeLocal, self.nMetabins),dtype=self.dtype) # the buffer to store local gradients for feature selection

    def compute_feature(self, codeLocalid, metabinid, normalize=True, target = None):
        '''
        compute the feature, and put it at self.featBuffer
        '''
        if target is None:
            target = self.featBuffer
        #self.featBuffer[:] = np.max(self.featSlice[codeLocalid, self.metabins[metabinid]], axis=0)
        # we have a faster method
        fm.fastmaxm(self.featSlice[codeLocalid], np.nonzero(self.metabins[metabinid])[0], target)
        if normalize:
            fm.normalizev(target, self.mLocal[codeLocalid, metabinid], self.stdLocal[codeLocalid, metabinid])
    
    def compute_feature_for_code(self, codeLocalid, normalize=True):
        '''
        compute all the features for codeLocalid and store them at featBufferPerCode
        '''
        for metabinid in range(self.nMetabins):
            self.compute_feature(codeLocalid, metabinid,normalize=normalize, target=self.featBufferPerCode[metabinid])

    def dump_current_state(self, filename):
        # dump the current state: feature mean, std, selected features, and the classifier
        # Only root will carry out this operation.
        all_m = self.comm.gather(self.mLocal, root = 0)
        all_std = self.comm.gather(self.stdLocal, root=0)
        if self.rank == 0:
            io.savemat(filename,{'m': np.vstack(all_m),\
                             'std': np.vstack(all_std),\
                             'nSelFeats': self.nSelFeats,\
                             'selCodeID': self.selCodeID,\
                             'selMetabinID':self.selMetabinID,\
                             'weights': self.weights,\
                             'b': self.b}, oned_as = 'row')

    def normalize_data(self, m = None, std = None):
        if self.normalized:
            mpi.rootprint('Warning: you are re-normalizing.')
        if m is None or std is None:
            # if either is none, we recompute.
            for i in range(self.nCodeLocal):
                for j in range(self.nMetabins):
                    self.compute_feature(i,j,normalize=False)
                    self.mLocal[i,j] = np.mean(self.featBuffer)
                    self.stdLocal[i,j] = np.std(self.featBuffer)+1e-8
        else:
            self.mLocal[:] = m
            self.stdLocal[:] = std
        self.normalized = True
            
    def load_data_batch(self, root, batch_size, file_template, labelfile, \
                        rootRead = True, isTest = False, \
                        local_cache_root = None, read_local_cache = False):
        '''
        load the data in batches. file_template should be 'filename_{}_{}.mat' 
        where the batch size and batch id will be filled. The mat file will 
        contain a variable called 'feat'. labelfile is the file for labels 
        starting from either 0 or 1 (our code converts the labels to 0 ~ nLabel-1).
        '''
        from scipy import io
        nBatches = int(np.ceil(float(self.nData) / batch_size))
        # deal both cases: batch starts with 0 or 1
        if os.path.exists(os.path.join(root,file_template.format(batch_size, 0))):
            allrange = range(nBatches)
        else:
            allrange = range(1,nBatches+1)
        if local_cache_root is not None and not os.path.exists(local_cache_root):
            try:
                os.makedirs(local_cache_root)
            except OSError:
                mpi.nodeprint('Warning: I cannot create the directory necessary.')
        if read_local_cache and local_cache_root is not None:
            # load from local cache
            sid = 0
            for bid in allrange:
                mpi.rootprint('From Local Cache: Loading batch {} of {}'.format(bid, nBatches))
                filename = os.path.join(local_cache_root, file_template.format(batch_size, bid))
                matdata = io.loadmat(filename)
                batchNdata = matdata['feat'].shape[2]
                self.featSlice[:,:,sid:sid+batchNdata] = matdata['feat']
                sid += batchNdata
        elif rootRead:
            # root reads the file, and then propagates the values to other machines
            dataid = 0 # current feature id
            dataBuffer = np.zeros(self.nBaseFeat, dtype = self.dtype)
            timer = Timer()
            for bid in allrange:
                mpi.rootprint('RootRead: Loading batch {} of {}'.format(bid, nBatches))
                if self.rank == 0:
                    # read only of I am root
                    filename = os.path.join(root, file_template.format(batch_size, bid))
                    print filename
                    matdata = io.loadmat(filename)
                    feat = matdata['feat']
                    batchNdata = feat.shape[0]
                else:
                    feat = None
                    batchNdata = 0
                # broadcast the features
                # it seems that doing this one-datum-by-one-datum is the fastest...
                batchNdata = self.comm.bcast(batchNdata, root=0)
                for batchfeatid in range(batchNdata):
                    if self.rank == 0:
                        dataBuffer[:] = feat[batchfeatid]
                    self.comm.Bcast(dataBuffer, root = 0)
                    # the data storage is like
                    # [bin1_code1 bin1_code2 ... bin1_codeK bin2_code1 ... binN_codeK]
                    # while our data is [nCodeLocal, nBins, nData]
                    self.featSlice[:,:,dataid] = \
                        dataBuffer.reshape(self.nBins, self.nCodes)[:,self.codeRange[0]:self.codeRange[1]].T
                    dataid += 1
                if local_cache_root is not None:
                    # write local cache, so we may read it back later
                    filename = os.path.join(local_cache_root, file_template.format(batch_size, bid))
                    try:
                        io.savemat(filename,{'feat': self.featSlice[:,:, dataid-batchNdata:dataid]}, oned_as='row')
                    except Exception, e:
                        mpi.nodeprint('Unable to save to local buffer {}'.format(filename))
                mpi.rootprint('Elapsed {} secs.'.format(timer.lap()))
        else:
            sid = 0
            # everyone for him/herself.
            for bid in allrange:
                mpi.nodeprint('Loading batch {} of {}'.format(bid,nBatches))
                filename = os.path.join(root, file_template.format(batch_size,bid))
                matdata = io.loadmat(filename)
                eid = sid + matdata['feat'].shape[0]
                # load the data into featSlice
                for i in range(sid,eid):
                    self.featSlice[:,:,i] = \
                        matdata['feat'][i-sid].reshape([self.nBins, self.nCodes])\
                        [:,self.codeRange[0]:self.codeRange[1]].T
                if local_cache_root is not None:
                    # write local cache, so we may read it back later
                    filename = os.path.join(local_cache_root, file_template.format(batch_size, bid))
                    try:
                        io.savemat(filename,{'feat': self.featSlice[:,:, sid:eid]}, oned_as='row')
                    except Exception, e:
                        mpi.nodeprint('Unable to save to local buffer {}'.format(filename))
                sid = eid
                # this sometimes helps python do garbage collection
                matdata = None
        # load label
        if self.rank == 0:
            matdata = io.loadmat(os.path.join(root, labelfile))
            # if the label starts with 1, make it start with 0
            if matdata['label'].min() == 1:
                matdata['label'] -= 1
            self.rawlabels[:] = matdata['label'].reshape(matdata['label'].size)[:self.nData]
            matdata = None
        self.comm.Bcast(self.rawlabels, root=0)
        for i in range(self.nData):
            # we need to make the label matrix a -1/1 matrix 
            self.labels[self.rawlabels[i],i] = 1
        if not isTest:
            mpi.rootprint('Normalizing training data')
            timer = Timer()
            self.normalize_data()
            mpi.nodeprint('Normalization took {} secs.'.format(timer.lap()))
           
    def append_feature(self,codeid, metabinid):
        '''
        find the owner of the feature, broadcast it to all the nodes, and append the 
        feature to the currently selected features if necessary.
        from the owner of the feature, broadcast this feature and append it to the 
        current selected features. Each instance will update the slice of data it 
        is responsible for
        '''
        # find the owner
        owner = int( codeid / self.ncode_per_node )
        if self.rank == owner:
            self.compute_feature(codeid-self.codeRange[0],metabinid)
        self.comm.Bcast(self.featBuffer,root=owner)
        if self.dataSel is not None:
            self.dataSel[self.nSelFeats] = self.featBuffer
        self.isSelected[codeid, metabinid] = True
        self.selCodeID[self.nSelFeats] = codeid
        self.selMetabinID[self.nSelFeats] = metabinid
        self.nSelFeats += 1
    
    def append_multiple_features(self, codeidlist, metabinidlist, reset=True):
        '''
        Append a set of features in idxlist to the selected Features.
        This can be used to test the performance of a set of pre-selected
        features.
        '''
        if reset:
            self.isSelected[:] = False
            self.nSelFeats = 0
        for i in range(len(codeidlist)):
            self.append_feature(codeidlist[i],metabinidlist[i])
        
    def select_new_feature_by_grad(self, samplePerRun):
        '''
        the routine to select a new feature by the gradient magnitude
        '''
        # compute the local gradient magnitude for feature selection
        # gL is a nData*nLabel matrix
        self.gL = np.ascontiguousarray(gL_bnll(self.labels, self.curr_wxb).T, dtype=self.dtype)
        
        if samplePerRun == 1:
            # this might take some time: for each feature, we basically
            # need to regenerate features and compute the dot.
            # to alleviate the problem a little bit, we regenerate a batch
            # every time
            for codeLocalid in range(self.nCodeLocal):
                self.compute_feature_for_code(codeLocalid, normalize=True)
                #self.localGradMat is a [self.nCodeLocal, self.nMetabins, self.nLabel] matrix
                self.localGradMat[codeLocalid] = np.dot(self.featBufferPerCode, self.gL)
            
            # for those features that are selected, we need to add their regularizers
            my_features = np.nonzero((self.selCodeID[:self.nSelFeats] >= self.codeRange[0]) & \
                          (self.selCodeID[:self.nSelFeats] < self.codeRange[1]))[0]
            for feat in my_features:
                curr_weight = self.weights[:,feat]
                self.localGradMat[self.selCodeID[feat]-self.codeRange[0],self.selMetabinID[feat]] \
                    += self.gamma * self.nData * curr_weight
            self.scoreVec[:] = np.sum(self.localGradMat**2, axis=2)
    
            local_opt_feat_id = self.scoreVec.argmax()
            local_opt_feat_codeid = local_opt_feat_id / self.nMetabins
            local_opt_feat_metabinid = local_opt_feat_id % self.nMetabins
            local_opt_feat_score = self.scoreVec[local_opt_feat_codeid, local_opt_feat_metabinid]
            local_opt_feat_codeid += self.codeRange[0]
        else:
            local_opt_feat_codeid = 0
            local_opt_feat_metabinid = 0
            local_opt_feat_score = 0.0
            sampleSize = np.int(self.nCodeLocal*self.nMetabins*samplePerRun)
            # get the not selected id
            not_selected_ones = np.nonzero(1-self.isSelected[self.codeRange[0]:self.codeRange[1]])
            if sampleSize > len(not_selected_ones[0]):
                sampleSize = len(not_selected_ones[0])
            # shuffle
            randlist = np.array(range(len(not_selected_ones[0])), dtype=np.int)
            np.random.shuffle(randlist)
            temp_feat_codelocalid = not_selected_ones[0][randlist]
            temp_feat_metabinid = not_selected_ones[1][randlist]
            # do things in batches
            batchsize = self.nMetabins
            temp_feat_score = np.zeros(batchsize, dtype=self.dtype)
            for i in range(np.int(np.ceil(sampleSize/np.float(batchsize)))):
                start = i*batchsize
                end = np.minimum(start+batchsize, sampleSize)
                for j in range(start,end):
                    # we will use featBufferPerCode to store the computed features
                    self.compute_feature(temp_feat_codelocalid[j], temp_feat_metabinid[j],\
                                         target=self.featBufferPerCode[j-start])
                temp_gradMat = np.dot(self.featBufferPerCode[:end-start], self.gL)
                temp_scoreVec = np.sum(temp_gradMat**2, axis=1)
                temp_opt_feat_id = temp_scoreVec.argmax()
                temp_opt_feat_score = temp_scoreVec[temp_opt_feat_id]
                if temp_opt_feat_score > local_opt_feat_score:
                    local_opt_feat_score = temp_opt_feat_score
                    local_opt_feat_codeid = temp_feat_codelocalid[temp_opt_feat_id+start]+self.codeRange[0]
                    local_opt_feat_metabinid = temp_feat_metabinid[temp_opt_feat_id+start]
        
        self.safebarrier()
        [opt_feat_score, opt_feat_codeid, opt_feat_metabinid] = self.comm.allreduce(\
            [local_opt_feat_score, local_opt_feat_codeid, local_opt_feat_metabinid], \
            op=MPI.MAX)
        
        return opt_feat_score, opt_feat_codeid, opt_feat_metabinid

    def retrain_model(self, nActiveSet=None, samplePerRun = 1.0, factr = 10, pgtol = 1e-08, iprint=-1):
        '''
        train the current model. Since we often have multiple labels, we will ask 
        each node to do one optimization
        '''
        loss = 0
        if nActiveSet is not None and np.abs(nActiveSet) < self.nSelFeats:
            # gradient computation will be carried out only when necessary
            gw_local = np.zeros((self.nLabel, self.nSelFeats), dtype = self.dtype)
            gw_reduced = np.zeros((self.nLabel, self.nSelFeats), dtype = self.dtype)
            my_features = np.nonzero((self.selCodeID[:self.nSelFeats] >= self.codeRange[0]) & \
                                     (self.selCodeID[:self.nSelFeats] < self.codeRange[1]))[0]
            
            if samplePerRun == 1:
                # distributed gradient computation - actually, lookup.
                for feat in my_features:
                    gw_local[:,feat] = self.localGradMat[self.selCodeID[feat]-self.codeRange[0], self.selMetabinID[feat]]
            else:
                # compute each one
                for feat in my_features:
                    self.compute_feature(self.selCodeID[feat]-self.codeRange[0], self.selMetabinID[feat])
                    gw_local[:,feat] = np.dot(self.featBuffer, self.gL) + \
                                       self.gamma * self.nData * self.weights[:,feat]
            # reduce gw_local so we all have the gradients. There will be only one non-zero element
            # over all the nodes so MPI.SUM will work.
            self.safebarrier()
            self.comm.Allreduce(gw_local, gw_reduced, op = MPI.SUM)
            
        # do approximate model retraining
        for idx in range(self.nLabel):
            if idx % self.size != self.rank:
                # I am not responsible for this label
                continue
            else:
                #if iprint >= 0:
                mpi.nodeprint('Training label {}/{}'.format(idx, self.nLabel))
                if nActiveSet is None or np.abs(nActiveSet) >= self.nSelFeats:
                    # retrain the whole model
                    weight_range = range(self.nSelFeats)
                    dataActiveSet = self.dataSel[:self.nSelFeats]
                else:
                    # we will retrain the model involving the previous 'nActiveSet' number
                    # of features.
                    # if nActiveSet == 0, this is equivalent to boosting 
                    # if nActiveSet > 0, we choose the latest features
                    # if nActiveSet < 0, we choose the features with the largest gradients
                    if nActiveSet >= 0:
                        weight_range = range(self.nSelFeats-nActiveSet, self.nSelFeats)
                    else:
                        # The last feature will always be selected
                        gw_idx = np.argsort(np.abs(gw_reduced[idx,:-1]))
                        weight_range = np.hstack((gw_idx[nActiveSet+1:], self.nSelFeats-1))
                    # create a contiguous storage to help memory access patterns
                    dataActiveSet = np.empty((len(weight_range), self.nData), dtype = self.dtype)
                    dataActiveSet[:] = self.dataSel[weight_range]
                # update curr_wxb
                if nActiveSet is None:
                    # completely retrain
                    self.curr_wxb[idx] = 0
                    self.b[idx] = 0
                    self.weights[idx] = 0
                else:
                    self.curr_wxb[idx] -= np.dot(self.weights[idx,weight_range], dataActiveSet)
                opt_return = optimize.fmin_l_bfgs_b(ObjFuncIncrement, \
                        np.hstack((self.weights[idx,weight_range], 1.0)),\
                        args = (dataActiveSet, \
                                self.labels[idx], \
                                self.curr_wxb[idx], \
                                self.gamma),\
                        factr = factr, pgtol = pgtol, iprint=iprint)
                wb = opt_return[0]
                self.weights[idx,weight_range] = wb[:-1]
                self.b[idx] += wb[-1]
                # update curr_wxb
                self.curr_wxb[idx] += np.dot(wb[:-1], dataActiveSet) + wb[-1]
                # accumulate loss
                loss += opt_return[1]
        # wait for all the nodes to finish their job
        self.safebarrier()
        # propagate the wxb values and classifier weights
        for idx in range(self.nLabel):
            self.comm.Bcast(self.curr_wxb[idx], root = idx % self.size)
            self.comm.Bcast(self.weights[idx], root = idx % self.size)
            self.b[idx] = self.comm.bcast(self.b[idx], root = idx % self.size)
        loss = self.comm.allreduce(loss, op=MPI.SUM)
        return loss
    
    def compute_current_accuracy(self):
        '''
        Using the current w'x+b to predict the accuracy. The label is simply determined
        as the one with the largest wxb value.
        ''' 
        return np.sum( np.argmax(self.curr_wxb, axis=0) == self.rawlabels ) / float(self.nData)
    
    def compute_test_accuracy(self, w, b, confMat = False):
        '''
        compute accuracy for test data
        '''
        if self.dataSel is None:
            # just a precausion: if I do not host data, forget about it.
            return 0.0
        predict = np.argmax(np.dot(w[:, :self.nSelFeats], self.dataSel[:self.nSelFeats]).T + b, axis=1)
        if confMat:
            confMat = np.zeros((self.nLabel,self.nLabel))
            for i in range(self.nLabel):
                for j in range(self.nLabel):
                    confMat[i,j] = np.sum((predict==i)&(self.rawlabels==j))
            return np.sum(predict==self.rawlabels)/float(self.nData), confMat
        else:
            return np.sum(predict == self.rawlabels) / float(self.nData)
            
    def restore_from_dump_file(self, filename, tester=None, dataOnly = False):
        print 'Not implemented yet.'
        '''
        # old grafting code:
        matdata = io.loadmat(filename)
        mpi.rootprint('Loading from dump file {}'.format(filename))
        nSelFeatsDump = matdata['nSelFeats'][0,0]
        selFeatIDDump = matdata['selFeatID'][0,:nSelFeatsDump]
        self.isSelected[:] = False
        # propagate data
        self.append_multiple_features(selFeatIDDump)
        if tester is not None:
            tester.append_multiple_features(selFeatIDDump)
        # debug code
        if self.nSelFeats != nSelFeatsDump:
            print 'Warning: {} != {}'.format(self.nSelFeats, nSelFeatsDump)
        
        if not dataOnly:
            self.weights[:,:nSelFeatsDump] = matdata['weights'][:,:nSelFeatsDump]
            self.b[:] = matdata['b'].reshape(self.nLabel)
            # curr_wxb will only be computed by root
            if self.rank == 0:
                self.curr_wxb = np.dot(self.weights[:,:self.nSelFeats].copy(), self.dataSel[:self.nSelFeats]) \
                                + self.b.reshape(self.nLabel,1)
            self.comm.Bcast(self.curr_wxb, root=0)
        '''

    def train_whole_model(self, tester=None):
        '''
        test the performance using all the features
        may be memory consuming.
        '''
        self.comm.barrier()
        mpi.rootprint('*'*46)
        mpi.rootprint('*'*15+'whole featureset'+'*'*15)
        mpi.rootprint('*'*46)
        
        if tester is not None:
            # normalize the test data with the stats of the training data
            tester.normalize_data(self.mLocal, self.stdLocal)
        
        timer = Timer()
        timer.reset()
        if self.maxGraftDim != self.nMetabins*self.nCodes:
            mpi.rootprint('Please initialize with maxGraftDim=nMetabins*nCodes')
            return
        self.nSelFeats = 0
        self.isSelected[:] = False
        mpi.rootprint('Generating Features...')
        for code in range(self.nCodes):
            for metabin in range(self.nMetabins):
                self.append_feature(code, metabin)
                if tester is not None:
                    tester.append_feature(code, metabin)
        mpi.rootprint('Feature generation took {} secs'.format(timer.lap()))
        mpi.rootprint('Training...')
        loss = self.retrain_model(None)
        mpi.rootprint('Training took {} secs'.format(timer.lap()))
        mpi.rootprint('Training accuracy: {}'.format(self.compute_current_accuracy()))
        if tester is not None:
            mpi.rootprint('Current Testing accuracy: {}'.format(tester.compute_test_accuracy(self.weights, self.b)))

    def randomselecttest(self, tester=None, random_iterations=1):
        '''
        test the performance of random selection
        '''
        self.comm.barrier()
        mpi.rootprint('*'*46)
        mpi.rootprint('*'*15+'random selection'+'*'*15)
        mpi.rootprint('*'*46)
        
        trainaccu = np.zeros(random_iterations)
        testaccu = np.zeros(random_iterations)
        
        if tester is not None:
            # normalize the test data with the stats of the training data
            tester.normalize_data(self.mLocal, self.stdLocal)
        
        itertimer = Timer()
        for iter in range(random_iterations):
            itertimer.reset()
            mpi.rootprint('*'*15+'Round {}'.format(iter)+'*'*15)
            if self.rank == 0:
                #decide which features we are going to select
                allidx = np.array(range(self.nCodes*self.nMetabins),dtype=np.int)
                np.random.shuffle(allidx)
                codeidlist = allidx / self.nMetabins
                metabinidlist = allidx % self.nMetabins
            else:
                codeidlist = None
                metabinidlist = None
            codeidlist = self.comm.bcast(codeidlist, root=0)
            metabinidlist = self.comm.bcast(metabinidlist, root=0)
            
            self.append_multiple_features(codeidlist[:self.maxGraftDim], metabinidlist[:self.maxGraftDim])
            mpi.rootprint('Feature selection took {} secs'.format(itertimer.lap()))
            mpi.rootprint('Training...')
            loss = self.retrain_model(None)
            trainaccu[iter] = self.compute_current_accuracy()
            mpi.rootprint('Training took {} secs'.format(itertimer.lap()))
            mpi.rootprint('Current training accuracy: {}'.format(trainaccu[iter]))
            if tester is not None:
                tester.append_multiple_features(codeidlist[:self.maxGraftDim], metabinidlist[:self.maxGraftDim])
                testaccu[iter] = tester.compute_test_accuracy(self.weights, self.b)
                mpi.rootprint('Current Testing accuracy: {}'.format(testaccu[iter]))
            mpi.rootprint('Testing selection took {} secs'.format(itertimer.lap()))
        self.safebarrier()
        
        mpi.rootprint('*'*15+'Summary'+'*'*15)
        mpi.rootprint('Training accuracy: {} +- {}'.format(np.mean(trainaccu),np.std(trainaccu)))
        mpi.rootprint('Testing accuracy: {} +- {}'.format(np.mean(testaccu),np.std(testaccu)))

    def graft(self, dump_every = 0, \
              dump_file = None, \
              nActiveSet = None, \
              tester = None, \
              test_every = 10, \
              samplePerRun = 1, \
              fromDumpFile = None \
             ):
        '''
        the main grafting algorithm
        ==Parameters==
        dump_every: the frequency to dump the current result. 0 if you do not want to dump
        dump_file: dump file name.
        nActiveSet: when retraining, the number of features in the active set.
            pass None for full retraining (may be slow!)
            pass a positive number to select the last features
            pass a negative number to select features via their gradient values
                (recommended, much better than other approaches)
            pass 0 for boosting
        tester: the grafterMPI class that hosts the test data
        test_every: the frequency to compute test accuracy
        samplePerRun: in each feature selection run, how many features (in proportions)
            we should sample to select feature from. Pass 1 to enumerate all features.
        fromDumpFile: restore from dump file (not implemented for the mb version yet)
        '''
        self.comm.barrier()
        mpi.rootprint('*'*38)
        mpi.rootprint('*'*15+'grafting'+'*'*15)
        mpi.rootprint('*'*38)

        if True:
            mpi.rootprint('Number of data: {}'.format(self.nData))
            mpi.rootprint('Number of labels: {}'.format(self.nLabel))
            mpi.rootprint('Number of codes: {}'.format(self.nCodes))
            mpi.rootprint('Bins: {0}x{0}'.format(self.nBinsPerEdge))
            mpi.rootprint('Total pooling areas: {}'.format(self.nMetabins))
            mpi.rootprint('Total features: {}'.format(self.nMetabins*self.nCodes))
            mpi.rootprint('Number of features to select: {}'.format(self.maxGraftDim))
        mpi.rootprint('Graft Settings:')
        mpi.rootprint('dump_every = {}\nnActiveSet={}\ntest_every={}\nsamplePerRun={}'.format(\
                            dump_every, nActiveSet, test_every, samplePerRun))
        self.comm.barrier()
        
        if tester is not None:
            # normalize the test data with the stats of the training data
            tester.normalize_data(self.mLocal, self.stdLocal)
        if fromDumpFile is not None:
            self.restore_from_dump_file(fromDumpFile, tester)
        
        old_loss = 1e10
        timer = Timer()
        itertimer = Timer()
        for T in range(self.nSelFeats, self.maxGraftDim):
            itertimer.reset()
            mpi.rootprint('*'*15+'Round {}'.format(T)+'*'*15)
            score, codeid, metabinid = self.select_new_feature_by_grad(samplePerRun)
            mpi.rootprint('Selected Feature [code: {}, metabin: {}], score {}'.format(codeid, metabinid, score))
            # add this feature to the selected features
            self.append_feature(codeid, metabinid)
            mpi.rootprint('Number of Features: {}'.format(self.nSelFeats))
            mpi.rootprint('Feature selection took {} secs'.format(itertimer.lap()))
            mpi.rootprint('Retraining the model...')
            loss = self.retrain_model(nActiveSet, samplePerRun)
            mpi.rootprint('Total loss reduction {}/{}={}'.format(loss, old_loss, loss/old_loss))
            mpi.rootprint('Current training accuracy: {}'.format(self.compute_current_accuracy()))
            mpi.rootprint('Model retraining took {} secs'.format(itertimer.lap()))
            old_loss = loss

            if tester is not None:
                tester.append_feature(codeid, metabinid)
                if (T+1) % test_every == 0:
                    # print test accuracy
                    test_accuracy = tester.compute_test_accuracy(self.weights, self.b)
                    mpi.rootprint('Current Testing accuracy: {}'.format(test_accuracy))
                    
            self.safebarrier()
            mpi.rootprint('This round took {} secs, total {} secs'.format(timer.lap(), timer.total()))
            mpi.rootprint('ETA {} secs.'.format(timer.total() * (self.maxGraftDim-T)/(T+1.0e-5)))
            
            if dump_every > 0 and (T+1) % dump_every == 0 and dump_file is not None:
                mpi.rootprint('*'*15 + 'Dumping' + '*'*15)
                self.dump_current_state(dump_file + str(T)+'.mat')
        
        mpi.rootprint('*'*15+'Finalizing'.format(T)+'*'*15)
        if dump_file is not None:
            self.dump_current_state(dump_file + 'final.mat')
        
if __name__ == "__main__":
    # Let's moo if it works.
    import utils.moo

