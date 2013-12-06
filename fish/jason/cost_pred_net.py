#! /usr/bin/env python

import theano
from theano import tensor
from theano.compat.python2x import OrderedDict
import theano.sparse
import warnings
import numpy.random
from theano.tensor.shared_randomstreams import RandomStreams

from pylearn2.costs.cost import Cost
from pylearn2.space import Conv2DSpace, Conv3DSpace
from pylearn2.utils.misc import Container

import ipdb as pdb

# JBY: copied from costs/autoencoder.py


class PredNetCostBase(Cost):
    '''
    Base class for PredNet costs
    '''

    supervised = False

    def cost_from_prediction(self, future, prediction):
        predErrors = prediction - future
        cost = (predErrors**2).mean()

        return cost
        
    def get_fn_cost_from_prediction(self):
        '''Compiles and returns a theano function f(future,prediction) that
        returns the cost of making the given prediction of the given future
        '''

        example_future_shape = (4, 2, 10, 10, 1)
        example_future = numpy.zeros(example_future_shape,
                                     dtype = theano.config.floatX)
        th_future = theano.tensor.TensorType(theano.config.floatX,[False]*5)('th_future')
        th_prediction = theano.tensor.TensorType(theano.config.floatX,[False]*5)('th_prediction')
        th_future.tag.test_value = example_future
        th_prediction.tag.test_value = example_future  # predict zeros too
        th_cost = self.cost_from_prediction(th_future, th_prediction)
        fn_cost_from_prediction = theano.function(
            [th_future, th_prediction],
            th_cost, name = 'fn_cost_from_prediction')
        return fn_cost_from_prediction
        
    def get_fn_expr(self, model):
        '''Compiles and returns a theano function f(data_batch) that
        returns the cost of the given data_batch using the supplied
        model.
        '''

        example_batch_size = 4
        example_batch_shape = ((example_batch_size,) +             # e.g. 4
                              model.input_space.shape +            # e.g. (3,10,10)
                              (model.input_space.num_channels,))   # e.g. 1
        example_batch = numpy.zeros(example_batch_shape,
                                    dtype = theano.config.floatX)
        th_batch = theano.tensor.TensorType(theano.config.floatX,[False]*5)('th_batch')
        th_batch.tag.test_value = example_batch
        th_cost = self.expr(model, th_batch)
        fn_expr = theano.function([th_batch], th_cost, name = 'fn_expr')
        return fn_expr

    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())



class PredNetCostMLP(PredNetCostBase):
    '''
    Cost for the MLP prediction version of PredNet
    '''

    #def expr(self, model, data, *args, **kwargs):
    def expr(self, model, data, ignore_mode = False):
        self.get_data_specs(model)[0].validate(data)
        if not ignore_mode:
            assert model.mlp_mode and not model.ae_mode, 'This cost is only for mlp mode'

        # TODO here: slice past from present, predict, compute avg error.

        # For now, assume we predict one time slice from the N-1 others

        # Input `data' comes in the format of input_space:
        #   input_space = Conv3DSpace((self.time_slices, self.window_size[0], self.window_size[1]),
        #                              num_channels = self.num_channels)

        # Ordering is: ('b', 0, 1, 2, 'c'), time is axis 0
        past = data[:,:-model.future_timeslices]
        future = data[:,-model.future_timeslices:]

        prediction = model.past_to_future(past)

        cost = self.cost_from_prediction(future, prediction)

        return cost



class PredNetCostAE(PredNetCostBase):
    '''
    Cost for AE version of PredNet
    '''

    def __init__(self, lambd = 1, recon_terms = -1, pred_terms = -1, walkback = False):
        '''Parameters:
        lambd : float
            relative weight of prediction term
        recon_terms : int
            number of reconstruction terms to consider (-1 to consider
            all). For a model with P past timesteps and F future
            timesteps, there are F + 1 possible recon_terms and F
            possible pred_terms.
        pred_terms : int
            number of prediction terms to consider (-1 to consider all). See above.
        walkback : bool
            if False, all time steps are predicted from the previous time slice
            if True, all time steps are predicted from the first time slice

        '''
        assert lambd >= 0, 'lambd must be non-negative'
        self.lambd = float(lambd)
        assert recon_terms >= -1, 'recon_terms must be -1, 0, or positive'
        self.recon_terms = recon_terms
        assert pred_terms >= -1, 'pred_terms must be -1, 0, or positive'
        self.pred_terms = pred_terms
        assert isinstance(walkback, bool), 'walkback should be a bool'
        self.walkback = walkback
    
    #def expr(self, model, data, *args, **kwargs):
    def expr(self, model, data, separate_costs = False):
        ####### different....
        self.get_data_specs(model)[0].validate(data)
        assert model.ae_mode and not model.mlp_mode, 'This cost is only for ae mode'
        recon_terms = self.recon_terms
        if recon_terms == -1:
            recon_terms = model.future_timeslices + 1
        assert recon_terms <= model.future_timeslices + 1, (
            'recon_terms is %d but model.future_timeslices is '
            '%d (too big)' % (recon_terms, model.future_timeslices))
        pred_terms = self.pred_terms
        if pred_terms == -1:
            pred_terms = model.future_timeslices
        assert pred_terms <= model.future_timeslices, (
            'pred_terms is %d but model.future_timeslices is '
            '%d (too big)' % (pred_terms, model.future_timeslices))

        # Reconstruction cost
        recon_costs = []
        for ii in range(recon_terms):
            data_ii = data[:,ii:ii+model.past_timeslices]
            # Keep this separate from encode below so that DAE will
            # add noise via corruptor. If the AE is not a DAE, this
            # computation will probably/hopefully be optimized away.
            reconstruction_ii = model.layers_ae.reconstruct(data_ii)
            recon_costs.append(((reconstruction_ii - data_ii) ** 2).mean())

        # Prediction cost
        # Compute all encodings, even if they won't be used (should be optimized away).
        encodings = []
        for ii in range(model.future_timeslices+1):
            # Produces 2 encodings for the 2 past -> 1 future model
            data_ii = data[:,ii:ii+model.past_timeslices]
            encoding_ii = model.layers_ae.encode(data_ii)
            encodings.append(encoding_ii)

        pred_costs = []
        if self.walkback:
            # Predict all futures from first encoding
            current_encoding = encodings[0]
            for ii in range(pred_terms):
                predicted_encoding = model.layers_pred.fprop(current_encoding)
                actual_encoding = encodings[ii+1]
                cost_pred.append(self.lambd * ((predicted_encoding - actual_encoding) ** 2).mean())
                current_encoding = predicted_encoding
        else:
            # Predict each future from previous encoding
            for ii in range(pred_terms):
                predicted_encoding = model.layers_pred.fprop(encodings[ii])
                actual_encoding = encodings[ii+1]
                pred_costs.append(((predicted_encoding - actual_encoding) ** 2).mean())

        cost_total = sum(recon_costs) + sum(pred_costs)

        if separate_costs:
            return recon_costs, pred_costs
        else:
            return cost_total

    def get_monitoring_channels(self, model, data, **kwargs):
        ret = super(PredNetCostAE, self).get_monitoring_channels(model, data, **kwargs)
        assert isinstance(ret, OrderedDict)

        recon_costs, pred_costs = self.expr(model, data, separate_costs = True)
        for ii in range(len(recon_costs)):
            ret['recon_cost_%i' % ii] = recon_costs[ii]
        for ii in range(len(pred_costs)):
            ret['pred_cost_%i' % ii] = pred_costs[ii]

        # Also track the associated cost of pixel-level predictions,
        # although this cost is not directly optimized by
        # PredNetCostAE!
        dummy_mlp_cost = PredNetCostMLP()
        ret['pixel_pred_cost'] = dummy_mlp_cost.expr(model, data, ignore_mode = True)

        return ret





# Just an alias (for old pickle files)
PredNetCost = PredNetCostMLP
