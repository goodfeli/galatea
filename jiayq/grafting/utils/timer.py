from time import time
from utils import mpi

def hms(t,template='{}h {}m {:.2f}s'):
    '''
    format the time value to "xxh xxm xxs"
    '''
    # not implemented
    hour = int(t / 3600.0)
    t = t % 3600.0
    minute = int (t / 60)
    t = t % 60
    return template.format(hour,minute,t)

class Timer:
    '''
    class Timer implements some sugar functions that works like a stopwatch.
    Timer.reset() resets the watch
    Timer.lap()   returns the time elapsed since the last lap() call
    Timer.total() returns the total time elapsed since the last reset
    '''
    def __init__(self):
        # t is the total time
        # l is the lap time
        self.t = time()
        self.l = time()
    
    def reset(self):
        self.t = time()
        self.l = time()
        
    def lap(self):
        diff = time() - self.l
        self.l = time()
        return diff
    
    def total(self):
        return time() - self.t

class LoopReporter:
    '''
    class LoopReporter implements some sugar functions that reports
    the stats of a loop that Yangqing often needs.
    '''
    def __init__(self, step = 100, header = '', rootOnly = False):
        self.timer = Timer()
        self.header = header
        self.step = step
        self.rootOnly = rootOnly
        
    def reset(self):
        self.timer.reset()

    def report(self,processed,total):
        if processed % self.step != 0:
            return
        elapsed = self.timer.total()
        if processed == 0:
            eta = 0.0
        else:
            eta = elapsed * (total - processed) / processed
        if self.rootOnly:
            mpi.rootprint('{} {}/{}, elapsed {}, eta {}.'.format(self.header, processed, total, hms(elapsed), hms(eta)))
        else:
            mpi.nodeprint('{} {}/{}, elapsed {}, eta {}.'.format(self.header, processed, total, hms(elapsed), hms(eta)))

