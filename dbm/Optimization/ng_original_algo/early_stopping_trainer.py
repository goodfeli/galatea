import numpy
import time, sys, os

class EarlyStoppingTrainer:
    def __init__(self, n_train_batches, n_valid_batches, n_test_batches, n_epoch=2000):

        self.n_train_batches = n_train_batches
        self.n_valid_batches = n_valid_batches
        self.n_test_batches = n_test_batches
        self.n_epoch = n_epoch
        
    def train(self, agent, learn_fn_scipy=False):
        """
        use scipy optimization when learn_fn_scipy is True
        """
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_params = None
        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        iter=1
        while (epoch < self.n_epoch) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):

                if learn_fn_scipy:
                    minibatch_avg_cost = agent.learn_fn_scipy(minibatch_index)
                else:
                    
                    #jacobian = agent.J_fn(minibatch_index)
                    #outputs = agent.outputs_fn(minibatch_index)
                    minibatch_avg_cost = agent.learn_fn(minibatch_index, 1.0/iter)
                    print iter
                    assert not numpy.isnan(minibatch_avg_cost)
                    #pass
                # iteration number
                #import pdb; pdb.set_trace()
                print minibatch_avg_cost
                #params = agent.params_fn()
                #grads = agent.grads_fn(minibatch_index)
                #J = agent.J_fn(minibatch_index)
                #G = agent.G_fn(minibatch_index)
                G_inv = agent.G_inv_fn()
                #delta_theta = agent.delta_theta_fn(minibatch_index)
                #import ipdb; ipdb.set_trace()
                print 'G_inv:', numpy.abs(G_inv).max()
                iter = epoch * self.n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [agent.valid_cost_fn(i) for i
                                         in xrange(self.n_valid_batches)]
                    
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f ' %
                         (epoch, minibatch_index + 1, self.n_train_batches,
                          this_validation_loss))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [agent.test_cost_fn(i) for i
                                       in xrange(self.n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f ') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score))

                if patience <= iter:
                        done_looping = True
                        break

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss, best_iter, test_score))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
