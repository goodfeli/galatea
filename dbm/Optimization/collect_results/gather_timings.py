import sys
import glob
import numpy
import matplotlib.pyplot as plt
import matplotlib.text as text

def main(data_dict):
    fig = plt.figure()
    color = ['g','b','r','c','m','y','k']
    line_style = ['-','--','-.','.']
    style = ['g-','b-','r-','c-','m-','y-','k-','g-.','b-.','r-.','c-.','m-.','y-.','k-.' ]
    
    legend = []
    style_index=0

    for id, timings in data_dict.items():
        #import pdb; pdb.set_trace()
        if '_sgd' in id and 'nat_sgd' not in id:
            continue
        train_stats = timings['train']
        train_cost = train_stats[:,11]
        epochs = range(train_cost.shape[0])
        seconds = numpy.sum(train_stats[:,[0,1,2]], axis=1)
        time_accum = numpy.add.accumulate(seconds)
            
        plt.plot(epochs, train_cost, style[style_index], label=id)
        #plt.plot(time_accum, train_cost, style[style_index], label=id)
        legend.append(id)
        style_index +=1
        """
        ax= fig.add_subplot(122)
        plt.plot(time_accum, train_cost, 'b')
        plt.grid(False)
        plt.xlabel('seconds --->')
        plt.ylabel('train cost --->')
        #plt.legend(('training cost vs epoch', 'training cost vs seconds'),
        #                      'upper center', shadow=True)
        #plt.ylim([0,50])
    
        plt.title(algo + ' on ' + data_name)
        """
    plt.legend(loc=1, prop={'size':8})
    plt.grid(False)
    plt.xlabel('epoch --->')
    #plt.xlabel('time --->')
    plt.ylabel('train cost --->')
    #plt.ylim([0,4])
    #plt.xlim([1,50])
    plt.show()

if __name__ == '__main__':
    import pdb; pdb.set_trace()

    files=[]
    for f in sys.argv[1:]:
        f = f + '/jobman_*/timing.npz'
        f = glob.glob(f)
        files.append(f)
    
    files = [item for sublist in files for item in sublist]
    


    # key is the id of the job
    # value is the timing.npz
    data_dict = {}
    for f in files:
        if f != []:
            path = f
            names = path.split('/')
            id = names[-4] + names[-2] + '_' + names[-3]
            timings = numpy.load(path)
            data_dict[id] = timings

    mnist={}
    cifar={}
    for key, value in data_dict.items():
        if 'mnist' in key:
            mnist[key]=value
        elif 'cifar' in key:
            cifar[key]=value
        else:
            raise NotImplementedError('unknow dataset in ' + key)
    import pdb; pdb.set_trace()

    main(mnist)
    


        
