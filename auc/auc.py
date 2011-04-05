import numpy as N

def auc(Output, Target, pos_small =  0, precise_ebar = 0, show_fig=0, dosigma = True):
    """area, sigma = auc(Output, Target, pos_small, precise_ebar, show_fig)
    # This is the algorithm proposed for
    # computing the AUC and the error bar.
    # It is assumed that the outputs provide a score
    # with the negative examples having the lowest score
    # unless the flag pos_small = 1.
    # precise_ebar=1: slower but better error bar calculation.
    """

    #print 'ENTER AUC'

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
        temp = N.zeros((Output.shape[0], 1) , Output.dtype)
        temp[:,0] = Output
        Output = temp

    area=[]
    sigma=[]
    n= Target.shape[0]
    negidx = N.nonzero(Target<0)[0] # indices of negative class elements
    posidx = N.nonzero(Target>0)[0] # indices of positive class elements
    neg = negidx.shape[0]    # number of negative class elements
    pos = posidx.shape[0]    # number of positive class elements

    #print 'neg = '+str(neg)+', pos = '+str(pos)
    #assert False

    if neg==0 or pos==0:
        print 'returning area = '+str(area)
        assert False

        return area, sigma

    uval = N.unique(Output)
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
        i = N.argsort(output)
        u = output[i]

        uval_ascending = N.unique(output)
        assert len(uval.shape) == 1
        uval = N.flipud(uval_ascending)

        # Test whether there are ties
        if uval.shape[0] == n:
            S = N.zeros( (n, ) )
            S[i] = N.arange(1,n+1)
        else:
            # Another speed-up trick (maybe not critical): test whether we have a whole bunch
            # of negative examples with the same output
            last_neg = N.nonzero(output==output.max())[0]
            other = N.setdiff1d(N.arange(n), last_neg)

            L = last_neg.shape[0]

            if L>1 and N.unique(output[other]).shape[0] == other.shape[0]:
                S = N.zeros( (n, ) )
                S[i] = N.arange(1,n+1)
                S[last_neg] = n-(L-1)/2
            else:
                # Average the ranks for the ties
                oldval = u[0]
                newval = u[0]
                R = N.arange(n).astype('float64')+1.0
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
                S = N.zeros( (len(i), ), R.dtype)
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
        SEN = (SS-RR)/pos
        assert kk == len(area)
        area.append(float(SEN.sum())/float(neg) )              # compute the AUC
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
            SEN = (SS-RR) / pos                          # compute approximate sensitivity
            SPE = 1-(N.arange(1,neg+1)-0.5)/neg  # compute approximate specificity
                                                         # (new 0.5 Dec 5 correction)

            if precise_ebar:
                # Calculate the "true" ROC (slow)
                uval.sort()
                sensitivity = N.zeros((uval.shape[0]+1, 1))
                specificity = N.zeros((uval.shape[0]+1, 1))
                sensitivity[1] = 0
                specificity[1] = neg
                for k in xrange(uval.shape[0]):
                    sensitivity[k+1] = sensitivity[k] + N.nonzero(output[posidx]==uval[k])[0].shape[0]
                    specificity[k+1] = specificity(k) - N.nonzero(output[negidx]==uval[k])[0].shape[0]
                sensitivity = sensitivity / pos
                specificity = specificity / neg
            else:
                sensitivity = SEN
                specificity = SPE
            two_BAC = sensitivity + specificity    # compute twice the balanced accuracy
            print two_BAC.shape
            assert False
            [u,k] = max(two_BAC)                   #find its max value
            sen = sensitivity[k]                    # and the corresponding sensitivity
            spe = specificity[k]                    # and specificity
            sigma[kk] = 0.5 * N.sqrt(sen*(1-sen)/ pos + spe*(1-spe)/ neg) # error bar estimate

            # Plot the results
            if show_fig:
                print "TODO: implement plot showing"
                """figure; bar(SPE, SEN); xlim([0,1]); ylim([0,1]); grid on
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

