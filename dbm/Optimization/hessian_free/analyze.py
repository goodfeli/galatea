import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as text

def perform_plot(train_stats, valid_stats):

    train_cost = train_stats[:,0]
    rho = train_stats[:,1]
    ridge = train_stats[:,2]
    time = train_stats[:,3]
    
    valid_cost = valid_stats[:,0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    import ipdb; ipdb.set_trace()
    epochs = range(train_cost.shape[0])
    
    plt.plot(epochs, train_cost, 'k--', epochs, valid_cost, 'k:')

    plt.legend(('training cost', 'validation cost'),
                          'upper center', shadow=True)
    #plt.ylim([-1,20])
    plt.grid(False)
    plt.xlabel('epoch --->')
    plt.ylabel('cost --->')
    plt.title('hessian free')

    plt.show()
    
if __name__ == '__main__':
    
    path = sys.argv[1]
    data = np.load(path)
    train_stats = data['train_stats']
    valid_stats = data['valid_stats']
    train_stats = train_stats[train_stats[:,0] != 0]
    valid_stats = valid_stats[valid_stats[:,0] != 0]
    
    import ipdb; ipdb.set_trace()
    perform_plot(train_stats, valid_stats)
    
    
    
