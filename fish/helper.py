#! /usr/bin/env python

import theano
from theano import tensor



def softmax_to_idsN(softmax):
    '''
    Compute numerical (not 1-hot) IDs given input features.

    Parameters
    ----------
    softmax : tensor_like
        Theano symbolic representing the softmax activations. Assumed to be
        a 4-tensor in the space self.input_space

    Returns
    -------
    ids : tensor
        Theano symbolic of IDs.
    '''

    idsN = tensor.argmax(softmax, axis=1)
    idsN.name = 'idsN'

    return idsN

def xysincos_to_xy(xysincos):
    '''
    Compute predicted xy position given output of linear sublayer, xysincos.

    Parameters
    ----------
    xysincos : tensor_like
        Theano symbolic representing output of the linear sublayer. Assumed to be
        a 4-tensor in the space self.input_space

    Returns
    -------
    xy_hat : tensor
        Theano symbolic of predicted xy location (2-tensor of shape (batch_size, 2))
    '''

    xy_hat = xysincos[:,0:2]      # xy is always first two columns for fish or spheres
    xy_hat.name = 'xy_hat'

    return xy_hat

def xysincos_to_angles(xysincos, is_fish = True):
    '''
    Compute predicted angles given output of linear sublayer, xysincos.

    Parameters
    ----------
    xysincos : tensor_like
        Theano symbolic representing output of the linear sublayer. Assumed to be
        a 4-tensor in the space self.input_space

    Returns
    -------
    angles : tensor
        Theano symbolic of predicted angles (2-tensor of shape (batch_size, ???))   =====
    '''

    sincos_hat = xysincos[:,2:]      # either 2 or 4 sincos values for fish or spheres
    sincos_hat.name = 'sincos_hat'

    if is_fish:
        angles_hat   = tensor.arctan2(sincos_hat[:,0], sincos_hat[:,1])
        angles_hat.name = 'angles_hat'
        return angles_hat
    else:
        angles_hat_0 = tensor.arctan2(sincos_hat[:,0], sincos_hat[:,1])
        angles_hat_1 = tensor.arctan2(sincos_hat[:,2], sincos_hat[:,3])
        angles_hat_0.name = 'angles_hat_0'
        angles_hat_1.name = 'angles_hat_1'
        return angles_hat_0, angles_hat_1

def ids_to_wiskott_id_accuracy(ids_hat, ids_true):
    '''
    Compute the accuracy of the IDs, as in franzius2008invariant-object-recognition.

    Parameters
    ----------
    ids_hat : tensor_like
    ids_true : tensor_like
        Theano symbolic representing the true numerical (not 1-hot)
        IDs. Assumed to be a 1-tensor. NOTE: this should be passed in with type floatX!
        Perhaps via:
            idsN_floatX = array(idsN_int, dtype=theano.config.floatX)

    Returns
    -------
    accuracy : scalar floatX
        Theano scalar accuracy, e.g. 1 is 100%, 0 is 0% accuracy.
    '''

    accuracy_tmp = tensor.eq(ids_hat, ids_true).mean()
    accuracy_tmp.name = 'accuracy_tmp'

    accuracy = tensor.cast(accuracy_tmp, theano.config.floatX)
    accuracy.name = 'accuracy'

    return accuracy

def xyhat_to_wiskott_xy_errors(xy_hat, xy_true):
    '''
    Compute the accuracy of x and y, as in franzius2008invariant-object-recognition.

    Parameters
    ----------
    xy_hat : tensor_like
        Theano symbolic representing the true xy position.
        Assumed to be a 2-tensor with shape (batch_size,2)
    xy_true : tensor_like
        similar
    
    Returns
    -------
    xy_errors : array of length 2
        array([x_err, y_err]). Errors are RMSE / coordinate range of 2.0
    '''

    xy_errors = tensor.sqrt(tensor.sqr(xy_hat-xy_true).mean(0)) / 2.0     # The range of values is 2.0, from -1 to 1
    xy_errors.name = 'xy_errors'

    return xy_errors

def angles_to_wiskott_angle_errors(angles_hat, sincos_true, is_fish = True):
    '''Compute the angles errors using arctan2 on the sin and cos
    values, as in franzius2008invariant-object-recognition.

    Parameters
    ----------
    feat : tensor_like
        Theano symbolic representing a batch of features. Assumed to be
        a 4-tensor in the space self.input_space
    sincos_true : tensor_like
        Theano symbolic representing the true sin and cos values.
        Assumed to be a 2-tensor with shape (batch_size,N), where
        N = 2 for fish and N = 4 for spheres.

    Returns
    -------
    errors : numpy vector of floatX, length 1 for fish or 2 for spheres
        Errors are RMSE in units of radians!! (Convert to degrees
        for more direct comparison to
        franzius2008invariant-object-recognition.
    '''

    if is_fish:
        angles_true = tensor.arctan2(sincos_true[:,0], sincos_true[:,1])
        angle_errors = angles_hat - angles_true
        angle_errors = tensor.minimum(angle_errors, 360 - angle_errors)
        angle_errors = tensor.minimum(angle_errors, 360 + angle_errors)
    else:
        angles_hat_0, angles_hat_1 = angles_hat
        raise Exception('not implemented yet')

    rms_angle_err = tensor.sqrt(tensor.sqr(angle_errors).mean(0))
    rms_angle_err.name = 'rms_angle_err'

    return rms_angle_err
