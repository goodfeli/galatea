import numpy
import scipy
import scipy.sparse
import sys
import tsne
import calc_tsne
import render
import time
import cPickle

def vectosparsematvalue(path,NBDIMS):
    """
    This function converts the unlabeled training data into a scipy
    sparse matrix and returns it.
    """
    print >> sys.stderr , "Read and converting data file: %s to a sparse matrix"%path 
    # We first count the number of line in the file
    f = open(path, 'r')
    i = f.readline()
    ct = 0
    while i!='':
        ct+=1
        i = f.readline()
    f.close()
    # We allocate and fill the sparse matrix as a lil_matrix for efficiency.
    NBEX = ct
    train = scipy.sparse.lil_matrix((NBEX,NBDIMS))
    f = open(path, 'r')
    i = f.readline()
    ct = 0
    next_print_percent = 0.1
    while i !='':
        if ct / float(NBEX) > next_print_percent:
            print >> sys.stderr , "\tRead %s %s of file"%(next_print_percent*100,'%')
            next_print_percent += 0.1
        i = i[:-1]
        i = list(i.split(' '))
        for j in i:
            if j!='':
                idx,dum,val = j.partition(':')
                train[ct,int(idx)] = val 
        i = f.readline()
        ct += 1
    print >> sys.stderr , "Data converted" 
    # We return a csr matrix for efficiency 
    # because we will later shuffle the rows.
    return train.tocsr()

def ReadLabelFile(LabelFile, category):
    """
    This function returns a list of label corresponding to the category (0<=int<=5)
    """
    f = open(LabelFile, 'r')
    i = f.readline()
    listlab = []
    while i != '':
        listlabtmp = i[:-1].split(' ')
        listlab += [int(listlabtmp[category])]
        i = f.readline()
    return listlab
 

def CreateTSne(pathdata,pathdataraw,pathlabel,NBDIMS,labellist,perplexity,used = 'C',PCA = False,init_dims = 50,loadb = False):
    if not loadb:
        sp = vectosparsematvalue(pathdata,NBDIMS)
        tim = time.time()
        if used != 'C':
            Y=tsne.tsne(sp.toarray(),2,perplexity = perplexity,use_pca = PCA,initial_dims = init_dims)
        else:
            Y = calc_tsne.calc_tsne(sp.toarray(),2,PERPLEX = perplexity, INITIAL_DIMS = 50,PCAb=PCA)
        f = open(pathdata[:-4] + '_tsne%s_%s_%s_%s.dat'%(perplexity,used,PCA,init_dims),'w')
        cPickle.dump(Y,f,-1)
    else:
        f = open(pathdata[:-4] + '_tsne%s_%s_%s_%s.dat'%(perplexity,used,PCA,init_dims),'r')
        Y = cPickle.load(f)
        f.close()
    sp = vectosparsematvalue(pathdataraw,NBDIMS)
    mask = numpy.exp(-numpy.arange(NBDIMS)/NBDIMS*5)
    listpoints = []
    for i in range(sp.shape[0]):
        vect = sp[i,:].toarray()
        listpoints += [('%s'%((vect>0).sum()),Y[i,0],Y[i,1])]
    for i in labellist:
        label = ReadLabelFile(pathlabel, i)
        listlab = []
        for j in range(len(label)):
            if label[j] == 1:
                listlab += ["#0ff"]
            if label[j] == 2:
                listlab += ["#0f0"]
            if label[j] == 3:
                listlab += ["#00f"]
            if label[j] == 4:
                listlab += ["#f00"]
            if label[j] == 5 or label[j] == -1:
                listlab += ["#000"]
        print len(listpoints), len(listlab)
        render.render(listpoints,pathdata[:-4] + '_tsne%s_lab%s_%s_%s_%s.png'%(perplexity,i,used,PCA,init_dims),label=listlab)
    print 'Elapsed time py :' , time.time() - tim, ' seconds'

if __name__ == '__main__':
    pathdata = sys.argv[1]
    pathdataraw = sys.argv[2]
    pathlabel = sys.argv[3]
    NBDIMS = int(sys.argv[4])
    labellist = eval(sys.argv[5])
    perplexity = float(sys.argv[6])
    used = sys.argv[7]
    PCA = eval(sys.argv[8])
    init_dims = int(sys.argv[9])
    loadb = eval(sys.argv[10])
    CreateTSne(pathdata,pathdataraw,pathlabel,NBDIMS,labellist,perplexity,used,PCA,init_dims,loadb)
