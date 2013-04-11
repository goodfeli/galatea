class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self,data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

import sys
sys.stdout=Unbuffered(sys.stdout)

# Generic imports
import numpy
import cPickle
import gzip
import time

from utils import print_mem, print_time

class MainLoop(object):
    def __init__(self, data, model, algo, state, channel):
        ###################
        # Step 0. Set parameters
        ###################
        self.data = data
        self.state = state
        self.channel = channel
        self.model = model
        self.algo = algo
        self.state['validcost'] = 1e20
        self.state['bvalidcost'] = 1e20
        self.state['testcost'] = 1e20
        self.state['traincost'] = 1e20
        self.state['validtime'] = 0
        self.state['btraincost'] = 1e20
        self.state['bvalidcost'] = 1e20
        n_elems = state['loopIters'] // state['trainFreq']
        self.timings = {}
        for name in self.algo.return_names:
            self.timings[name] = numpy.zeros((n_elems,), dtype='float32')
        n_elems = state['loopIters'] // state['validFreq'] + 1
        self.timings['valid'] = numpy.zeros((n_elems,), dtype='float32')
        self.timings['test'] = numpy.zeros((n_elems,), dtype='float32')
        if self.channel is not None:
            self.channel.save()
        self.start_time = time.time()
        self.batch_start_time = time.time()

    def validate(self):
        cost = self.model.validate()
        print ('** validation cost %6.3f computed in %s'
               ', best cost is %6.3f, test %6.3f, whole time %6.3f min') % (
                   cost,
                   print_time(time.time() - self.batch_start_time),
                   self.state['bvalidcost'],
                   self.state['testcost'],
                   (time.time() - self.start_time)/60. )
        self.batch_start_time = time.time()
        pos = self.step // self.state['validFreq']
        self.timings['valid'][pos] = float(cost)
        self.timings['test'][pos] = float(self.state['testcost'])
        self.state['validcost'] = float(cost)
        self.state['validtime'] = float(time.time() - self.start_time)/60.
        if self.state['bvalidcost'] > cost:
            self.state['bvalidcost'] = float(cost)
            self.state['btraincost'] = float(self.state['traincost'])
            self.test()
        print_mem('validate')

    def test(self):
        self.model.best_params = [(x.name, x.get_value()) for x in
                                  self.model.params]
        cost = self.model.test_eval()
        print '>>> Test cost', cost
        pos = self.step // self.state['validFreq']
        self.timings['test'][pos] = float(cost)
        self.state['testcost'] = float(cost)

    def save(self):
        numpy.savez(self.state['prefix']+'timing.npz',
                    **self.timings)
        if self.state['overwrite']:
            self.model.save(self.state['prefix']+'model.pkl')
        else:
            self.model.save(self.state['prefix']+'model%d.pkl' % self.save_iter)
        cPickle.dump(self.state, open(self.state['prefix']+'state.pkl', 'w'))
        self.save_iter += 1

    def main(self):
        print_mem('start')
        self.state['gotNaN'] = 0
        self.start_time = time.time()
        self.batch_start_time = time.time()
        self.step = 0
        self.save_iter = 0
        self.save()
        if self.channel is not None:
            self.channel.save()
        self.save_time = time.time()
        last_cost = 1.
        start_time = time.time()
        self.start_time = start_time
        while self.step < self.state['loopIters'] and \
              last_cost > .1*self.state['minerr'] and \
              (time.time() - start_time)/60. < self.state['timeStop']:
            if (time.time() - self.save_time)/60. > self.state['saveFreq']:
                self.save()
                if self.channel is not None:
                    self.channel.save()
                self.save_time = time.time()
            st = time.time()
            try:
                rvals = self.algo()
                self.state['traincost'] = float(rvals['cost'])
                self.state['step'] = self.step
                last_cost = rvals['cost']
                for name in rvals.keys():
                    pos = self.step // self.state['trainFreq']
                    self.timings[name][pos] = rvals[name]

                if numpy.isinf(rvals['cost']) or numpy.isnan(rvals['cost']):
                    self.state['gotNaN'] = 1
                    self.save()
                    if self.channel:
                        self.channel.save()
                    print 'Got NaN while training'
                    last_cost = 0
                if self.step % self.state['validFreq'] == 0:
                    self.validate()
                self.step += 1
            except:
                self.state['wholetime'] = float(time.time() - start_time)
                self.save()
                if self.channel:
                    self.channel.save()

                last_cost = 0
                print 'Error in running natgrad (lr issue)'
                print 'BEST SCORE'
                print 'Validation', self.state['validcost']
                print 'Validation time', print_time(self.state['validtime'])
                print 'Train cost', self.state['traincost']
                print 'Best Train', self.state['btraincost']
                print 'Best Valid', self.state['bvalidcost']
                print 'TEST', self.state['testcost']
                print 'Took', (time.time() - start_time)/60.,'min'
                raise

        self.state['wholetime'] = float(time.time() - start_time)
        self.validate()
        self.save()
        if self.channel:
            self.channel.save()
        print 'BEST SCORE'
        print 'Validation', self.state['validcost']
        print 'Validation time', print_time(self.state['validtime'])
        print 'Train cost', self.state['traincost']
        print 'Best Train', self.state['btraincost']
        print 'Best Valid', self.state['bvalidcost']
        print 'TEST', self.state['testcost']
        print 'Took', (time.time() - start_time)/60.,'min'
