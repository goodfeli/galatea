#! /usr/bin/env python

from pylearn2.models.mlp import CompositeLayer, Layer
from pylearn2.utils import wraps
import ipdb as pdb

from theano import tensor

from fish.helper import (softmax_to_idsN, xysincos_to_xy,
                         xysincos_to_angles, ids_to_wiskott_id_accuracy,
                         xyhat_to_wiskott_xy_errors, angles_to_wiskott_angle_errors)



class FishyCompositeLayer(CompositeLayer):
    '''
    Extension of CompositeLayer that reports other error metrics.
    '''

    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):
        '''Reports Wiskott metrics....'''

        # First, get super class monitors, if any
        channels = super(FishyCompositeLayer, self).get_monitoring_channels_from_state(state, target=target)

        # Then get channels from each component
        if target is None:
            target = [None] * len(state)
        for ii,layer in enumerate(self.layers):
            channel_dict = layer.get_monitoring_channels_from_state(state[ii], target=target[ii])
            for key,val in channel_dict.iteritems():
                channels['%s_%s' % (layer.layer_name, key)] = val

        softmax_ids1H_hat = state[0]
        xysincos_hat = state[1]
        softmax_ids1H_true = target[0]
        xysincos_true = target[1]

        idsN_hat = softmax_to_idsN(softmax_ids1H_hat)
        idsN_true = softmax_to_idsN(softmax_ids1H_true)
        channels['wiskott_id_accuracy'] = ids_to_wiskott_id_accuracy(idsN_hat, idsN_true)
        xyerrs = xyhat_to_wiskott_xy_errors(xysincos_hat[:,:2], xysincos_true[:,:2])
        channels['wiskott_err_x'] = xyerrs[0]
        channels['wiskott_err_y'] = xyerrs[1]

        is_fish = True
        assert is_fish, 'not generalized yet! Need to get is_fish bool from somewhere...'
        sincos_hat  = xysincos_hat[:,2:]
        sincos_true = xysincos_true[:,2:]
        if is_fish:
            angles_hat = tensor.arctan2(sincos_hat[:,0], sincos_hat[:,1])
        else:
            angles_hat = (tensor.arctan2(sincos_hat[:,0], sincos_hat[:,1]),
                          tensor.arctan2(sincos_hat[:,2], sincos_hat[:,3]))
        channels['wiskott_err_angle'] = angles_to_wiskott_angle_errors(angles_hat, sincos_true, is_fish = is_fish)

        return channels
