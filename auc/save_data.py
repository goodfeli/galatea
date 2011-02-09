def save_data(directory, name, data):
    for set_name in data.X0:
        if set_name == 'devel':    #do not save and send development data
            continue
        #
    
        X = data.X0[set_name]
        try:
            X = X.todense()
        except AttributeError:
            pass
        #
        
        fname = name + '_' + set_name + '.prepro'
        filename = directory + '/' + fname

        print '==> Saving '+ fname

        if str(X.dtype).find('int') != -1 or data.quantized:
            save_int(filename, X)
        else:
            save(filename, 'X', '-ascii')
        #
        print 'done'
        
    #
#


def save_int(filename, X):
    f = open(filename, 'w')
    for i in xrange(X.shape[0]):
        for j in xrange(X.shape[1]):
            f.write(str(int(X[i,j]))+' ')
        #
        f.write('\n')
    #
    f.close()
#    

