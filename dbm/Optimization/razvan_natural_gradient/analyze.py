import sys
import numpy
import matplotlib.pyplot as plt
import matplotlib.text as text
"""
for the reference
self.train_timing[self.k, 0] = rvals['time_grads']
self.train_timing[self.k, 1] = rvals['time_metric']
self.train_timing[self.k, 2] = rvals['time_eval']
self.train_timing[self.k, 3] = rvals['score']
self.train_timing[self.k, 4] = rvals['minres_iters']
self.train_timing[self.k, 5] = rvals['minres_relres']
self.train_timing[self.k, 6] = rvals['minres_Anorm']
self.train_timing[self.k, 7] = rvals['minres_Acond']
self.train_timing[self.k, 8] = rvals['grad_norm']
self.train_timing[self.k, 9] = rvals['beta']
self.train_timing[self.k, 10] = rvals['lambda']
self.train_timing[self.k, 11] = rvals['error']
self.train_timing[self.k, 12] = rvals['time_err']
self.train_timing[self.k, 13] = rvals['damping']
self.train_timing[self.k, 14] = rvals['rho']

self.valid_timing: k,score
self.test_timing: k,score
"""
def perform_plot(train_stats, algo, data_name):

    train_cost = train_stats[:,11]
        
    epochs = range(train_cost.shape[0])
    seconds = numpy.sum(train_stats[:,[0,1,2]], axis=1)
    time_accum = numpy.add.accumulate(seconds)

    fig = plt.figure()
    ax = fig.add_subplot(121)
        
    plt.plot(epochs, train_cost, 'k')
    

    #plt.legend(('training cost vs epoch', 'training cost vs seconds'),
    #                      'upper center', shadow=True)
    #plt.ylim([0,50])
    plt.grid(False)
    plt.xlabel('epoch --->')
    plt.ylabel('train cost --->')
    

    ax= fig.add_subplot(122)
    plt.plot(time_accum, train_cost, 'b')
    #plt.legend(('training cost vs epoch', 'training cost vs seconds'),
    #                      'upper center', shadow=True)
    #plt.ylim([0,50])
    plt.grid(False)
    plt.xlabel('seconds --->')
    plt.ylabel('train cost --->')
    plt.title(algo + ' on ' + data_name)
    
    plt.show()
    
if __name__ == '__main__':
    
    path = sys.argv[1]
    algo = sys.argv[2]
    data_name = sys.argv[3]
    data = numpy.load(path)

    train_stats = data['train']
    valid_stats = data['valid']
    
        
    perform_plot(train_stats, algo, data_name)
