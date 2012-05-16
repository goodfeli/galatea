'''
MPI implements common util functions 
based on mpi4py.
'''

import numpy as np
import time
from mpi4py import MPI
import socket
import random
import os

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host_raw = socket.gethostname()
# this is the hack that removes things like ".icsi.berkeley.edu"
if host_raw.find('.') == -1:
    host = host_raw
else:
    host = host_raw[:host_raw.find('.')]
MPI_PRINT_MESSAGE_TAG=560710

random.seed(time.time()*rank)

def mkdir(dirname):
    '''
    make a directory. Avoid race conditions.
    '''
    try:
        os.makedirs(dirname)
    except OSError:
        pass
    
def vote():
    '''
    vote() randomly chooses a node from all the nodes.
    Input:
        None
    Output:
        a random number between 1 and mpi.size
    '''
    president = 0
    if rank == 0:
        president = random.randrange(0,size)
    president = comm.bcast(president, root=0)
    return president

def isRoot():
    '''
    returns if the current node is root
    '''
    return rank==0

def rootprint(string, newline = True):
    '''
    print only if I am root
    '''
    # used in MPI: only print if I am root.
    if rank == 0:
        if newline:
            print string
        else:
            print string,

def report():
    '''
    write out a short report for mpi.
    '''
    rootprint('***MPI starting***')
    rootprint('root: {}'.format(host))
    rootprint('size: {}'.format(size))
    rootprint('******************')
    safebarrier()

def collectprint(string):
    '''
    each node sends the message to root, and root prints out all messages.
    This function guarantees that the output is in the order of nodes.
    '''
    # first, send the messages
    comm.send('Node {:3d} {}: {}'.format(rank, host, string), dest=0, tag=MPI_PRINT_MESSAGE_TAG)
    if rank==0:
        for i in range(size):
            recvstr = comm.recv(source=i, tag=MPI_PRINT_MESSAGE_TAG)
            print recvstr
    safebarrier()
        
def nodeprint(string):
    '''
    each node prints. This function differs from collectprint: it does not synchronize things.
    '''
    print 'Node {:2d}-{} {}'.format(rank, host, string)

def safebarrier(tag=0, sleep=0.01):
    '''
    This is a better mpi barrier than MPI.comm.barrier(): the original barrier
    may cause idle processes to still occupy the CPU, while this barrier waits.
    '''
    if size == 1: 
        return 
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

class DataSeparator:
    def __init__(self,total):
        if type(total) is not int:
            # if a list ps passed, get the length, and record the guy
            self._initlist = total
            total = len(total)
        self._total = total
        starts = [int(total*i/size) for i in range(size)]
        self._start = starts[rank]
        ends = [int(total*i/size) for i in range(1,size+1)]
        self._end = ends[rank]
        self._local = ends[rank] - starts[rank]
        self._owner = []
        for i in range(size):
            self._owner += [i] * (ends[i] - starts[i])

    def __call__(self,l = None):
        '''
        returns a generator that goes through the local
        elements of the list l
        
        if l is None, we go through the list which is passed to the initializer
        '''
        if l is None:
            for i in range(self._start,self._end):
                yield self._initlist[i]
        else:
            for i in range(self._start,self._end):
                yield l[i]
    
    def total(self):
        return self._total

    def local(self):
        return self._local

    def owner(self,i):
        return self._owner[i]
    
    def start(self):
        return self._start

    def end(self):
        return self._end

    def collectMatrix(self,source,rootonly=True):
        '''
        collects a bunch of matrix that is computed by the
        distribution defined in the class. Each row of the
        matrix is a feature vector.
        '''
        dim = source.shape[1]
        if not rootonly:
            retMat = np.zeros((self._total, dim), dtype = source.dtype)
            # copy the local part
            retMat[self._start:self._end] = source
            for i in range(self._total):
                comm.Bcast(retMat[i],root=self._owner[i])
        else:
            if rank == 0:
                retMat = np.zeros((self._total, dim), dtype = source.dtype)
                # copy the local part
                retMat[self._start:self._end] = source
            else:
                retMat = None
            for i in range(self._total):
                if self._owner[i] == 0:
                    continue
                elif self._owner[i] == rank:
                    # send mine
                    comm.Send(source[i-self._start],dest=0)
                elif rank == 0:
                    comm.Recv(retMat[i], source=self._owner[i])
        return retMat

    def collectVector(self,source):
        '''
        Collects a bunch of vectors, similar to collectMatrix
        '''
        retVec = comm.allgather(source)
        return np.hstack(retVec)

if __name__ == "__main__":
    report()
