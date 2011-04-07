import sys

from numpy import *
from scipy.stats import pearsonr

def u_minus_s(a):
    return mean(a) - std(a)

def quantile(n):
    def f(a):
        a2 = sort(a)
        i = int(round(a2.shape[0] * n) / 100.0)

        if i < 0:
            i = 0
        elif i >= a2.shape[0]:
            i = a2.shape[0] - 1
            
        return a2[i]

    return f

def train4(a):
    return a[-1]

def train2(i):
    def f(a):
        return mean(a[4*i:4*(i+1)])
    return f

def stats(data, criterias):
    exps = unique(data[:,0])
    res = zeros((exps.shape[0], len(criterias)))
    #err = zeros((exps.shape[0], len(criterias)))

    for exp_idx in exps:
        for i,criteria in enumerate(criterias):
            res[exp_idx,i] = criteria(data[data[:,0] == exp_idx,3]) 

    return res
    
def extract_alcs(data):
    exps = unique(data[:,0])
    alcs = zeros((exps.shape[0], 2))

    for exp_idx in exps:
        alcs[exp_idx, 0] = data[data[:,0] == exp_idx,-2][0]
        alcs[exp_idx, 1] = data[data[:,0] == exp_idx,-1][0]
    return alcs

def mobile_corr(x,y,window=3):
    r = zeros((x.shape[0] - window + 1))

    for i in range(x.shape[0] - window + 1):
        c, _ = pearsonr(x[i:i+window], y[i:i+window])
        r[i] = c
    return r
    

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        data = loadtxt(sys.argv[1])
    else:
        data = loadtxt('results.txt')

    if len(sys.argv) >= 3:
        outfile = sys.argv[2]
    else:
        outfile = "stats.txt"

    criterias = [mean, median, u_minus_s, quantile(25), train4] +\
                [train2(i) for i in range(6)]

    res = stats(data, criterias)

        

    savetxt(outfile, hstack((alcs,res))) 
