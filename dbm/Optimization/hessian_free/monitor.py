import numpy
import time

EPOCH_MAX=2000

class Monitor:
    def initialize(self, deep_autoencoder, valid_fn, test_fn, loopsIters):
        self.deep_autoencoder = deep_autoencoder
        # quadratic approximation loop
        self.big_loop_counter = 0
        # LCG loop
        self.small_loop_counter = 0
        """
        # obj_value, rho, ridge, time
        self.train_stats = numpy.zeros((EPOCH_MAX,4)) * -1

        # cost, time
        self.valid_stats = numpy.zeros((EPOCH_MAX,2)) * -1
        """
        self.best_valid_cost = -numpy.inf
        self.valid_fn = valid_fn
        self.test_fn = test_fn

        self.train_timing = numpy.zeros((loopsIters,
                                    15), dtype='float32')
        self.valid_timing = numpy.zeros((loopsIters,), dtype='float32')
        self.test_timing = numpy.zeros((loopsIters,), dtype='float32')

    def init_jobman_channel(self, channel, state):
        self.channel = channel
        #self.state = state
        #self.state['validscore'] = numpy.inf
        #self.state['testscore'] = numpy.inf
        
    def record_train_step(self, iter, obj_value, cost_value, rho, ridge, time):
        self.train_timing[iter , 0] = time
        self.train_timing[iter , 1] = 0
        self.train_timing[iter  , 2] = 0
        self.train_timing[iter  , 3] = cost_value
        self.train_timing[iter  , 4] = 0
        self.train_timing[iter  , 5] = 0
        self.train_timing[iter  , 6] = 0
        self.train_timing[iter  , 7] = 0
        self.train_timing[iter  , 8] = 0
        self.train_timing[iter  , 9] = 0
        self.train_timing[iter  , 10] = ridge
        self.train_timing[iter  , 11] = obj_value
        self.train_timing[iter  , 12] = 0
        self.train_timing[iter  , 13] = ridge
        self.train_timing[iter  , 14] = rho
        
    def record_valid_step(self, iter):
        #t = time.time()
        cost = numpy.mean(self.valid_fn())
        #t = time.time()
        self.valid_timing[iter] = cost
        
        #print 'validation cost at iteration %d is %f '%(iter, cost)
        #f cost > self.best_valid_cost:
        #   self.save_model()
        #if cost < self.best_valid_cost:
        #    self.best_valid_cost = cost
        #    self.state['validscore']=cost
        #    self.state['testscore']=cost
            
        self.channel.save()
        self.save_stats(iter)

    def record_test_error(self):
        pass

    def save_model(self):
        self.deep_autoencoder.save_rbm_weights('best_rbm_weights.pkl')
        self.deep_autoencoder.save_reconstruction_weights('best_reconstruction_weights.pkl')

    def save_stats(self, iter):
        # note that valid timing is the same as test timing since valid set is the same as test set.
        numpy.savez('timing.npz', train=self.train_timing,
                    valid=self.valid_timing, test=self.valid_timing,k=iter)
exp_monitor = Monitor() 
        
        