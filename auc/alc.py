import numpy as N

def alc(x, y):
    """ Compute the Area under the Learning Curve (ALC)
    % Inputs:
    % x --              Values of number of examples used for training
    % y --              Corresponding AUC values
    % Returns:
    % global_score --   Normalized ALC"""

    # Remove the first point (0 examples)
    if x[0] != 0:
        raise Exception('1st ex. is not 0')
    #

    rand_predict = y[0]
    x = x[1:]
    y = y[1:]

    # Create a log2 scale
    x = N.log2(x);

    # Compute the score
    A=0;
    # Integrate by the trapeze method
    for k in xrange(1,len(x)):
        deltax = x[k]-x[k-1]
        mu = (y[k]+y[k-1])/2.
        A += deltax * mu
    #

    # Normalize the score
    Arand = rand_predict *x[-1]
    Amax = x[-1]

    global_score = (A-Arand)/(Amax-Arand)

    assert not N.isnan(global_score)

    return global_score
""

