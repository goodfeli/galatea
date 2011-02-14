import cPickle, numpy, scipy, sys
import pylearn.io.filetensor
from pylearn.datasets.utlc import load_sparse_dataset, load_ndarray_dataset

dataset = ["avicenna", "harry", "rita", "sylvester", "terry", "ule"]

# Since we have a very small number of values in each set (max ~ 1000), we
# compute the uniformization with a bin-packing technique, this gives us a very
# small memory footprint

def compute_rank(data, max=999, skip_zeros=False, count_once=False):
    """ Construct an ndarray of size (max+1,#features) where we store the rank
        of each value in a features """

    rank = numpy.zeros(((max+1), data[0].shape[1]))

    # find the occurence of each value for each feature
    print "Computing count"
    for dataset in data:
        for j in range(dataset.shape[1]):
            for i in range(dataset.shape[0]):
                rank[dataset[i,j],j] += 1

    # compute the rank
    print "Computing rank"
    for j in range(rank.shape[1]):
        # number of element per features must be recounted at each row
        # if skip_zeros is set
        count = sum([d.shape[0] for d in data])
        if skip_zeros:
            count -= rank[0,j]
            rank[0,j] = 0
        rank[0][j] /= count
        for i in range(rank.shape[0]-1):
            rank[i+1,j] = (rank[i,j] + rank[i+1,j]/count)

    return rank

def compute_rank_sparse(data, max=999, count_once=False):
    """ Construct an ndarray of size (max+1,#features) where we store the rank
        of each value in a features, where the input is provided as a scipy.sparse
        format. Zeros are not counted.
    """

    rank = numpy.zeros(((max+1), data[0].shape[1]))

    # find the occurence of each value for each feature
    print "Computing count"
    for dataset in data:
        I,J,V = scipy.sparse.find(dataset)
        for i in range(len(I)):
            rank[V[i],J[i]] += 1

    # compute the rank
    print "Computing rank"
    for j in range(rank.shape[1]):
        count = sum(rank[:,j])
        # division by 0!
        if count == 0:
            count = 1
        rank[0][j] /= count
        for i in range(rank.shape[0]-1):
            rank[i+1,j] = (rank[i,j] + rank[i+1,j]/count)
            
    
    return rank


data = sys.argv[1]
loader = sys.argv[2]
if len(sys.argv) != 3 or (data not in dataset) or (loader not in ["dense","sparse"]):
    print "Usage: %s dataset type"
    print " where dataset is either [avicenna, harry, rita, sylvester, terry, ule] "
    print " and type is either [dense, sparse]"


if loader == "dense":
    loader = load_ndarray_dataset
    ranker = compute_rank
else:
    loader = load_sparse_dataset
    ranker = compute_rank_sparse

print "Processing %s"%(data)
train, valid, test = loader(data, normalize=False)

# uniformize on the 3 sets
rank = ranker([train,valid,test])

# convert
train = train.astype(float)
valid = valid.astype(float)
test  = test.astype(float)


for (set,name) in [(train,"train"), (valid,"valid"), (test,"test")]:
    if loader == "dense":
        for i in range(set.shape[0]):
            for j in range(set.shape[1]):
                set[i,j] = rank[set[i,j],j]
        pylearn.io.filetensor.write(open(data+'_'+name+'.ft','wb'),set)
    else:
        I,J,V = scipy.sparse.find(set)
        for i in range(len(I)):
            V[i] = rank[V[i],J[i]]

        cPickle.dump(set, open(data+'_'+name+'.npy','wb'))

