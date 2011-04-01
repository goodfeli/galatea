import numpy

def auc(Output, Target, pos_small =  0, precise_ebar = 0, show_fig=0, dosigma = True):
    """area, sigma = auc(Output, Target, pos_small, precise_ebar, show_fig)
    # This is the algorithm proposed for
    # computing the AUC and the error bar.
    # It is assumed that the outputs provide a score
    # with the negative examples having the lowest score
    # unless the flag pos_small = 1.
    # precise_ebar=1: slower but better error bar calculation.
    """

    #print 'EnumpyTER AUC'

    #print 'Output'
    #print Output.sum()
    #print 'Target'
    #print Target.sum()
    #print 'pos_small'
    #print pos_small
    #print 'precise_ebar'
    #print precise_ebar


    #assert False

    #print Target
    #die


    #print Output
    #die

    if len(Output.shape) == 1:
        temp = numpy.zeros((Output.shape[0], 1) , Output.dtype)
        temp[:,0] = Output
        Output = temp
    area=[]
    sigma=[]
    n= Target.shape[0]
    negidx = numpy.nonzero(Target<0)[0] # indices of negative class elements
    posidx = numpy.nonzero(Target>0)[0] # indices of positive class elements
    neg = negidx.shape[0]    # number of negative class elements
    pos = posidx.shape[0]    # number of positive class elements

    #print 'neg = '+str(neg)+', pos = '+str(pos)
    #assert False

    if neg==0 or pos==0:
        print 'returning area = '+str(area)
        assert False

        return area, sigma
    uval = numpy.unique(Output)
    if (not show_fig) and uval.shape[0] == 2 and uval.min() ==-1 and uval.max() ==1:
        # TODO: What's bac?
        area, sigma = bac(Output, Target);
        print 'returning area = '+str(area)
        assert False
        return area, sigma
    # This is hard to vectorize, we just loop if multiple columns for outputs
    nn,pp = Output.shape
    p = 1

    if nn != 1 and pp != 1:
        p=pp
    elif nn==1:
        Output = Output.T
        Target = Target.T

    for kk in xrange(p):
        output = Output[:,kk]
        if not pos_small:
            output = -output
        #temp = list(output)
        #temp = zip(temp,range(len(temp)))
        #temp = sorted(temp)
        #u,i = zip(*temp)
        
        # sort outputs, best come first (u=sorted vals, i=index)
            i = numpy.argsort(output)
            u = output[i]

        uval_ascending = numpy.unique(output)
        assert len(uval.shape) == 1
        uval = numpy.flipud(uval_ascending)

        # Test whether there are ties
        if uval.shape[0] == n:
            S = numpy.zeros( (n, ) )
            S[i] = numpy.arange(1,n+1)
        else:
            # Another speed-up trick (maybe not critical): test whether we have a whole bunch
            # of negative examples with the same output
            last_neg = numpy.nonzero(output==output.max())[0]
            other = numpy.setdiff1d(numpy.arange(n), last_neg)

            L = last_neg.shape[0]

            if L>1 and numpy.unique(output[other]).shape[0] == other.shape[0]:
                S[i] = numpy.arange(1,n+1)
                S[last_neg] = n-(L-1)/2
            else:
                # Average the ranks for the ties
                oldval = u[0]
                newval = u[0]
                R = numpy.arange(n).astype('float64')+1.0
                k0 = 0.0
                for k in xrange(1,n): #yes, 1, the matlab was a 2
                    newval=u[k]
                    #print ('k',k,'newval',newval)
                    if newval == oldval:
                        # moving average
                        val = R[k-1]*float(k-k0)/float(k-k0+1.0)+R[k]/float(k-k0+1.0)
                        #print ('k0',k0, 'k',k, 'val', val)
                        #if k0 == 2:
                        #    die
                        R[k0:k+1] = val
                    else:
                        #print 'updated k0 to '+str(k)
                        k0=float(k)
                    #
                    oldval = newval
                #
                #print 'R'
                #print R
                #die
                S = numpy.zeros( (len(i), ), R.dtype)
                for ridx, sidx in enumerate(i):
                    S[sidx] =  R[ridx]
                #print 'case L==1 or whatever'
                #print 'first three elements of S'
                #print S[0:3]
                #print 'first three elements of i'
                #print i[0:3]
                #print 'R indexing'
                #print (R[i[0]], R[i[1]], R[i[2]])
                #die

        SS = S[negidx]
        SS.sort()
        RR = range(neg)
        SEnumpy = (SS-RR)/pos
        assert kk == len(area)
        area.append(float(SEnumpy.sum())/float(neg) )              # compute the AUC
        #%%%%%%%%%%%%%%%%%%%%% ERROR BARS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if dosigma:
            # Adjust RR for the ties (new Dec 5 correction)
            oldval = SS[0]
            newval = SS[0]
            k0 = 0
            j = 1
            for k in xrange(1, len(SS) ):
                newval = SS[k]
                if newval == oldval:
                    # number of tied values
                    nt = k - k0 + 1
                    # moving average
                    RR[k0:k] = RR[k-1] * (k-k0)/nt + RR[k]/nt
                else:
                    k0=k
                    j = j+1
                oldval = newval
            SEnumpy = (SS-RR) / pos                          # compute approximate sensitivity
            SPE = 1-(numpy.arange(1,neg+1)-0.5)/neg  # compute approximate specificity
                                                         # (new 0.5 Dec 5 correction)

            if precise_ebar:
                # Calculate the "true" ROC (slow)
                uval.sort()
                sensitivity = numpy.zeros((uval.shape[0]+1, 1))
                specificity = numpy.zeros((uval.shape[0]+1, 1))
                sensitivity[1] = 0
                specificity[1] = neg
                for k in xrange(uval.shape[0]):
                    sensitivity[k+1] = sensitivity[k] + numpy.nonzero(output[posidx]==uval[k])[0].shape[0]
                    specificity[k+1] = specificity(k) - numpy.nonzero(output[negidx]==uval[k])[0].shape[0]
                sensitivity = sensitivity / pos
                specificity = specificity / neg
            else:
                sensitivity = SEnumpy
                specificity = SPE
            two_BAC = sensitivity + specificity    # compute twice the balanced accuracy
            print two_BAC.shape
            assert False
            [u,k] = max(two_BAC)                   #find its max value
            sen = sensitivity[k]                    # and the corresponding sensitivity
            spe = specificity[k]                    # and specificity
            sigma[kk] = 0.5 * numpy.sqrt(sen*(1-sen)/ pos + spe*(1-spe)/ neg) # error bar estimate

            # Plot the results
            if show_fig:
                print "TODO: implement plot showing"
                """figure; bar(SPE, SEnumpy); xlim([0,1]); ylim([0,1]); grid on
                xlabel('Specificity'); ylabel('Sensitivity');
                hold on; plot(specificity, sensitivity, 'ro'); plot(specificity, sensitivity, 'r-', 'LineWidth', 2);
                title(['AUC=' num2str(area) '+-' num2str(sigma)]); """

    if type(area) == type([]):
        assert len(area) == 1
        area = area[0]

    if type(area) != type(1.0):
        print area
        assert False

    return area, sigma


def alc(x, y):
    """ Compute the Area under the Learning Curve (ALC)
    % Inputs:
    % x --              Values of number of examples used for training
    % y --              Corresponding AUC values
    % Returns:
    % global_score --   numpyormalized ALC"""

    # Remove the first point (0 examples)
    if x[0] != 0:
        raise Exception('1st ex. is not 0')
    #

    rand_predict = y[0]
    x = x[1:]
    y = y[1:]

    # Create a log2 scale
    x = numpy.log2(x);

    # Compute the score
    A=0;
    # Integrate by the trapeze method
    for k in xrange(1,len(x)):
        deltax = x[k]-x[k-1]
        mu = (y[k]+y[k-1])/2.
        A += deltax * mu
    #

    # numpyormalize the score
    Arand = rand_predict *x[-1]
    Amax = x[-1]

    global_score = (A-Arand)/(Amax-Arand)

    assert not numpy.isnan(global_score)

    return global_score


def hebbian_learner(pos, neg, valid_prop=0.1):
    """
    pos: positive examples matrix (valid)
    neg: negative examples matrix (test)
    """
    rng = numpy.random.mtrand.RandomState(0)

    pos_inds = range(pos.shape[0])
    neg_inds = range(neg.shape[0])
    rng.shuffle(pos_inds)
    rng.shuffle(neg_inds)
    
    pos_train_size = int((1 - valid_prop) * len(pos_inds))
    neg_train_size = int((1 - valid_prop) * len(neg_inds))
    
    train_pos_inds = pos_inds[:pos_train_size]
    train_neg_inds = neg_inds[:neg_train_size]
    valid_pos_inds = pos_inds[pos_train_size:]
    valid_neg_inds = neg_inds[neg_train_size:]
    
    x = range(7)
    y = []
    for n_examples in x:
        n_examples = int(numpy.exp(n_examples))
        weights = pos[train_pos_inds[:n_examples]].mean(0) - neg[train_neg_inds[:n_examples]].mean(0)
        weights = numpy.atleast_2d(weights).T
        
        targets = numpy.append(numpy.ones(pos[valid_pos_inds].shape[0]), -1*numpy.ones(neg[valid_neg_inds].shape[0]))
        valid_x = numpy.append(pos[valid_pos_inds], neg[valid_neg_inds], axis=0)
        valid_y = numpy.dot(valid_x, weights).flatten()
        
        y.append(auc(valid_y, targets, dosigma=False)[0])
    
    return alc(x, y)
