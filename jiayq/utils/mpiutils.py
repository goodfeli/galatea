import numpy as np
import time
from mpi4py import MPI
import socket
import random
import numpy as np

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()
MPI_PRINT_MESSAGE_TAG=560710

random.seed(time.time()*rank)
    
def vote():
    president = 0
    if rank == 0:
        president = random.randrange(0,size)
    president = comm.bcast(president, root=0)
    return president

def isRoot():
    return rank==0

def rootprint(string, newline = True):
    # used in MPI: only print if I am root.
    if rank == 0:
        if newline:
            print string
        else:
            print string,

def report():
    rootprint('***MPI starting***')
    rootprint('root: {}'.format(host))
    rootprint('size: {}'.format(size))
    rootprint('******************')
    safebarrier()

def collectprint(string):
    # first, send the messages
    comm.send('Node {:3d} {}: {}'.format(rank, host, string), dest=0, tag=MPI_PRINT_MESSAGE_TAG)
    if rank==0:
        for i in range(size):
            recvstr = comm.recv(source=i, tag=MPI_PRINT_MESSAGE_TAG)
            print recvstr
    comm.barrier()
        
def nodeprint(string):
    print 'Node {:3d} {}: {}'.format(rank, host, string)

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


