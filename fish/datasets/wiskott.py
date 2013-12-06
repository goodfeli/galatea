#!/usr/bin/env python

import functools
import warnings
import os, cPickle, logging
_logger = logging.getLogger(__name__)

import ipdb as pdb
import numpy as np

import theano

from pylearn2.datasets import dense_design_matrix, Dataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix, DefaultViewConverter
from pylearn2.datasets.temporal_dense_design_matrix import TemporalDenseDesignMatrix, CopyingConcatenatingIterator
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import image
from pylearn2.space import CompositeSpace, Conv3DSpace, Conv2DSpace

####################
# JBY: Based on cifar10.py
####################

class WiskottVideoConfig(object):
    '''This is just a container for specifications for a WiskottVideo
    dataset (so that one can easily create train/valid/test datasets
    with identical config using YAML anchors.
    '''

    def __init__(self, axes = ('b', 0, 1, 2, 'c'),
                 num_frames = 3, height = 32, width = 32, num_channels = 1,
                 num_balls = 1, ball_diameter = 10,
                 vel_multiplier = 1.0, vel_noise = 0,
                 vel_decay = 0, frames_burnin = 1000,
                 frames_simulate = 10000, dtype = None,
                 min_val = 0, max_val = 1):

        # Arbitrary choice: we do the validation here, not in WiskottVideo
        assert isinstance(axes, tuple), 'axes must be a tuple'
        self.axes = axes

        assert num_frames > 0, 'num_frames must be positive'
        self.num_frames = num_frames
        assert height > 0, 'height must be positive'
        self.height = height
        assert width > 0, 'width must be positive'
        self.width = width
        assert num_channels == 1, 'only 1 channel is supported for now'
        self.num_channels = num_channels

        assert num_balls >= 0, 'num_balls must be non-negative'
        self.num_balls = num_balls
        assert ball_diameter > 0, 'ball_diameter must be positive'
        assert ball_diameter < self.height, 'ball_diameter must be < height'
        assert ball_diameter < self.width, 'ball_diameter must be < width'
        self.ball_diameter = ball_diameter
        assert vel_multiplier >= 0, 'vel_multiplier must be non-negative'
        assert vel_multiplier < 25, 'vel_multiplier should not be too large (else there will be a lot of recursion)'
        assert vel_noise >= 0, 'vel_noise must be non-negative'
        self.vel_noise = vel_noise
        assert vel_decay >= 0, 'vel_decay must be non-negative'
        self.vel_decay = vel_decay

        assert frames_burnin >= 0, 'frames_burnin must be >= 0'
        self.frames_burnin = frames_burnin
        assert frames_simulate >= self.num_frames, 'frames_simulate must be >= num_frames (preferably much more)'
        self.frames_simulate = frames_simulate

        #self._output_space = Conv3DSpace(
        #    (self.num_frames, self.height, self.width),
        #    num_channels = self.num_channels,            
        #)

        assert max_val > min_val, 'max_val must be > min_val'
        self.min_val = min_val
        self.max_val = max_val
        if dtype is None:
            dtype = theano.config.floatX
        self.dtype = dtype


class WiskottVideo(Dataset):
    '''Simulated single ball bouncing around a rectangular chamber
    with (optional) noise added each time step. This dataset is
    generated on the fly.
    '''

    _default_seed = (17, 2, 946)
    
    def __init__(self, which_set, config):
        '''Create a WiskottVideo instance'''
        
        assert which_set in ('train', 'valid', 'test')
        self.which_set = which_set

        # Copy main config from provided config
        self.axes            = config.axes
        self.num_frames      = config.num_frames
        self.height          = config.height
        self.width           = config.width
        self.num_channels    = config.num_channels
        self.num_balls       = config.num_balls
        self.ball_diameter   = config.ball_diameter
        self.vel_noise       = config.vel_noise
        self.vel_decay       = config.vel_decay
        self.frames_burnin   = config.frames_burnin
        self.frames_simulate = config.frames_simulate
        self.min_val         = config.min_val
        self.max_val         = config.max_val
        self.dtype           = config.dtype
        
        # Load data here...
        seedmap = {
            'train': 123,
            'valid': 456,
            'test': 789
        }

        # Generate dataset here
        # TODO: cache to disk and reload
        video = self._simulate(frames_simulate = self.frames_simulate,
                               rng_seed = seedmap[self.which_set],
                               burn_in = self.frames_burnin)
        #video_promoted = np.reshape(video, (1,) + video.shape)  # batch size of 1

        # Init TemporalDenseDesignMatrix
        #view_converter = DefaultViewConverter((self.height, self.width, self.num_channels),
        #                                      axes = ('b', 0, 1, 'c'))  # maybe??
        self._dense_design_matrix = DenseDesignMatrix(topo_view = video)
        
    def _simulate(self, frames_simulate, rng_seed = None, burn_in = 0):
        rng = np.random.RandomState(rng_seed)

        # New sequential
        chamber = Chamber(height = self.height,
                          width  = self.width,
                          num_balls = self.num_balls,
                          ball_diameter = self.ball_diameter,
                          vel_noise = self.vel_noise,
                          vel_decay = self.vel_decay,
                          rng = rng)

        res_shape = (frames_simulate, self.height, self.width, self.num_channels)
        buf = np.zeros(res_shape, dtype = bool)

        assert burn_in >= 0, 'burn_in must be non-negative'
        for frame_idx in xrange(burn_in):
            chamber.step()
        for frame_idx in xrange(frames_simulate):
            chamber.render_to(buf[frame_idx, :, :, 0])
            chamber.step()

        # Convert from bool to appropriate float type and scale
        result = np.array(buf, dtype=self.dtype)
        result *= self.max_val - self.min_val
        result += self.min_val

        return result

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False, ignore_data_specs=False):

        if batch_size is None: batch_size = 100
        if num_batches is None: num_batches = 100
        assert batch_size > 0
        assert num_batches > 0
        assert topo is None
        assert targets is None

        if mode is None: mode = 'shuffled_sequential'
        assert mode in ('sequential', 'shuffled_sequential'), (
            'Mode must be one of: sequential, shuffled_sequential'
        )
        if mode != 'shuffled_sequential':
            warnings.warn('billiard dataset returning its only supported iterator type -- shuffled -- despite the request to the contrary')
        if not ignore_data_specs:
            assert data_specs != None, 'Must provide data_specs'
            assert len(data_specs) == 2, 'data_specs must include only one tuple for "features"'
            assert type(data_specs[0]) is CompositeSpace, 'must be composite space...??'
            assert data_specs[0].num_components == 1, 'must only have one component, features'
            assert data_specs[1][0] == 'features', (
                'data_specs must include only one tuple for "features"'
            )
        
        #underlying_dataspecs = (self._output_space, 'features')
        underlying_space = Conv2DSpace((self.height, self.width),
                                       num_channels = self.num_channels)
        underlying_dataspecs = (underlying_space, 'features')

        self._underlying_iterator = self._dense_design_matrix.iterator(
            mode = 'random_slice',     # IMPORTANT: to return contiguous slices representing chunks of time!
            batch_size = self.num_frames,
            num_batches = num_batches * batch_size,
            rng=rng,
            data_specs=underlying_dataspecs,
            return_tuple=False
        )

        #pdb.set_trace()
        
        return CopyingConcatenatingIterator(
            self._underlying_iterator,
            num_concat = batch_size,
            return_tuple = return_tuple
        )



class Chamber(object):
    '''Represents the state of a ball bouncing in a chamber'''

    _default_seed = (17, 2, 946)

    def __init__(self, width = 32, height = 32, num_balls = 1,
                 ball_diameter = 10, start_pos = None, start_vel = None,
                 rng = None, vel_multiplier = 1.0, vel_noise = 0,
                 vel_decay = 0):
        '''
        Parameters
        ----------
        start_pos : tuple
           The starting (x,y) position of the ball, or None to
           randomly generate a starting position
        '''
        
        assert height > 0, 'height must be positive'
        assert width > 0, 'width must be positive'
        self.bounds = (height, width)
        assert num_balls >= 0, 'num_balls must be non-negative'
        assert num_balls <= 1, 'num_balls must == 1 (for now)'
        self.num_balls = num_balls
        assert ball_diameter > 0, 'ball_diameter must be positive'
        assert ball_diameter < self.bounds[0], 'ball_diameter must be < height'
        assert ball_diameter < self.bounds[1], 'ball_diameter must be < width'
        self.ball_diameter = float(ball_diameter)
        self.ball_radius = self.ball_diameter / 2
        assert vel_noise >= 0, 'vel_noise must be non-negative'
        self.vel_noise = vel_noise
        assert vel_decay >= 0, 'vel_decay must be non-negative'
        self.vel_decay = vel_decay

        # Only generate once
        self.ii_mg, self.jj_mg = np.meshgrid(range(self.bounds[0]),
                                             range(self.bounds[1]),
                                             indexing='ij')
        
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)
        #print '  Chamber.rng is', self.rng #, self.rng.get_state()

        if start_pos != None:
            assert isinstance(start_pos, tuple)
            assert len(start_pos) == 2
            position = np.array([float(start_pos[0]),
                                 float(start_pos[1])])
        else:
            validWidth = np.array([self.bounds[0] - self.ball_diameter,
                                   self.bounds[1] - self.ball_diameter])
            position = (rng.uniform(0, 1, 2) * validWidth
                        + self.ball_radius)
        if start_vel != None:
            assert isinstance(start_vel, tuple)
            assert len(start_vel) == 2
            velocity = np.array([float(start_vel[0]),
                                 float(start_vel[1])])
        else:
            velocity = rng.normal(0, 1, 2) * vel_multiplier

        assert self.is_valid_position(position)
        self.position = position
        self.velocity = velocity
        #print 'Created chamber:', self
    
    def validate_position(self):
        '''Validate that the given position is valid and represents a
        ball completely within the chamber. Position is in (i,j)
        coordinate system.'''
        if not self.is_valid_position(self.position):
            raise Exception('Invalid position')
        
    def is_valid_position(self, position):
        return (position[0] >= self.ball_radius and
                position[0] <= self.bounds[0] - self.ball_radius and
                position[1] >= self.ball_radius and
                position[1] <= self.bounds[1] - self.ball_radius)

    def step(self, step_size = 1.0, add_vel_noise = True):
        '''Take one timestep, possibly bouncing'''
        #print 'fix this'
        # TODO: fix this!!!

        self.validate_position()
        if add_vel_noise:
            self.velocity += self.rng.normal(0,1,2) * self.vel_noise
            self.velocity -= self.velocity * self.vel_decay
        self.position, self.velocity = self._compute_step_result(self.position, self.velocity, step_size)
        self.validate_position()

    def _compute_step_result(self, pos, vel, step_size, assert_one_step = False):
        '''Iterative, includes bounces'''

        assert self.is_valid_position(pos)

        # Try whole step
        proposal = pos + step_size * vel
        if self.is_valid_position(proposal):
            pos = proposal
        else:
            if assert_one_step:
                raise Exception('Expected single step to succeed but it failed.')
            # figure out which constraint was violated first
            assert np.abs(vel).max() > 0
            cur_max_step = step_size
            which_vel = None
            if abs(vel[0]) > 0:
                # positive direction
                this_max = (self.bounds[0] - self.ball_radius - pos[0]) / vel[0]
                if this_max > 0 and this_max < cur_max_step:
                    cur_max_step = this_max
                    which_vel = 0
                # negative direction
                this_max = (pos[0] - self.ball_radius) / -vel[0]
                if this_max > 0 and this_max < cur_max_step:
                    cur_max_step = this_max
                    which_vel = 0
            if abs(vel[1]) > 0:
                # positive direction
                this_max = (self.bounds[1] - self.ball_radius - pos[1]) / vel[1]
                if this_max > 0 and this_max < cur_max_step:
                    cur_max_step = this_max
                    which_vel = 1
                # negative direction
                this_max = (pos[1] - self.ball_radius) / -vel[1]
                if this_max > 0 and this_max < cur_max_step:
                    cur_max_step = this_max
                    which_vel = 1
            #if abs(vel[1]) > 0:
            #    dir_ii = max((self.bounds[1] - self.ball_radius - pos[1]) / vel[1],
            #                 (pos[1] - self.ball_radius) / - vel[1]))
            #    max_step_size = min(max_step_size, dir_ii)
            assert cur_max_step > 0
            assert cur_max_step < step_size
            assert which_vel != None
            # Take partial step, then change velocity, then take rest of step
            pos, vel = self._compute_step_result(pos, vel, cur_max_step, assert_one_step = True)
            if which_vel == 0:
                vel[0] = -vel[0]
            else:
                vel[1] = -vel[1]
            pos, vel = self._compute_step_result(pos, vel, step_size - cur_max_step)  # May take multiple iterations
            
        assert self.is_valid_position(pos)
        return pos, vel

    def render(self, dtype = None):
        '''Return '''

        if dtype == None:
            dtype = theano.config.floatX
        buf = zeros((self.height, self.width), dtype = dtype)

        self.render_to(buf)
        
        return buf
        
    def render_to(self, buf):
        '''Renders directly to the supplied buffer (which must be the right size)'''

        buf[:] = ((self.ii_mg - self.position[0])**2
                  + (self.jj_mg - self.position[1])**2 < self.ball_radius**2)

    def __str__(self):
        params = (self.bounds[0], self.bounds[1],
                  self.position[0], self.position[1],
                  self.velocity[0], self.velocity[1])
        return 'Chamber(%dx%d, pos = (%g,%g), vel = (%g,%g))' % params



def demo():
    num_frames = 300
    height = 10
    width  = 10

    config = WiskottVideoConfig(
        num_frames = num_frames,
        height = height,
        width  = width,
        ball_diameter = 5,
        min_val = .1,
        max_val = .9,
        vel_multiplier = 4.0,
        vel_noise = .04,
        vel_decay = .001,
        frames_burnin = 1000,
        frames_simulate = 20000,
        )

    bil = WiskottVideo('train', config)

    iterator = bil.iterator(mode = None,
                            batch_size = 10,
                            num_batches = 2,
                            ignore_data_specs = True)

    for batch_idx, batch in enumerate(iterator):
        #print batch.shape
        #print batch.sum()
        assert len(batch.shape) == 5
        assert batch.shape[4] == 1   # single channel

        if batch_idx == 0:
            prefix = 'billiard_frame_'
            for frame_idx in xrange(num_frames):
                frameIm = image.tile_raster_images(batch[0,frame_idx:frame_idx+1], img_shape = (height,width), tile_shape=(1,1))
                image.save('%s%03d.png' % (prefix, frame_idx), frameIm)
            print 'Saved: %s*.png' % prefix
            print 'Convert to gif by running:\n  convert -delay 3 %s* billiard_video.gif' % (prefix)

        flattened = np.reshape(batch,
                               (batch.shape[0]*batch.shape[1],
                                np.prod(batch.shape[2:])))

        tiled = image.tile_raster_images(-flattened,
                                         img_shape=[height, width],
                                         tile_shape=[batch.shape[0], batch.shape[1]],
                                         tile_spacing=(5,1))
        filename = 'billiard_batch_%03d.png' % batch_idx
        image.save(filename, tiled)
        #image.save(filename, 255-tiled)
        print 'Saved:', filename



if __name__ == '__main__':
    demo()
