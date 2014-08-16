from pylearn2.config import yaml_parse

mlp = yaml_parse.load("""!obj:pylearn2.models.mlp.MLP {
            layers: [
                     !obj:pylearn2.models.mlp.RectifiedLinear {
                         layer_name: 'gh0',
                         dim: 8000,
                         irange: .05,
                         #max_col_norm: 1.9365,
                     },
                     !obj:pylearn2.models.mlp.CompositeLayer {
                         layer_name: "composite",
                         layers: [
                            !obj:pylearn2.models.mlp.MLP {
                                layer_name: "conv_subnet",
                                layers: [
                                     !obj:pylearn2.models.mlp.Sigmoid {
                                         layer_name: 'conv_subnet_h0',
                                         dim: 8112,
                                         irange: .05,
                                         #max_col_norm: 1.9365,
                                     },
                                     !obj:pylearn2.models.mlp.SpaceConverter {
                                         layer_name: 'conv_subnet_converter',
                                         output_space: !obj:pylearn2.space.Conv2DSpace {
                                        shape: [13, 13],
                                        num_channels: 48,
                                        axes: ['c', 0, 1, 'b'],
                                    }},
                                     !obj:galatea.adversarial.deconv.Deconv {
                                     #W_lr_scale: .05,
                                     #b_lr_scale: .05,
                                         num_channels: 3,
                                         output_stride: [3, 3],
                                         kernel_shape: [8, 8],
                                         pad_out: 8,
                                         #max_kernel_norm: 1.9365,
                                         # init_bias: !obj:pylearn2.models.dbm.init_sigmoid_bias_from_marginals { dataset: *train},
                                         layer_name: 'conv_subnet_y',
                                         irange: .05,
                                         tied_b: 0
                                     }
                                 ] # end conv_subnet_layers
                             }, # end conv_subnet
                             !obj:pylearn2.models.mlp.MLP {
                                 layer_name: "fc_subnet",
                                 layers: [
                                     !obj:pylearn2.models.mlp.Sigmoid {
                                         layer_name: 'fc_subnet_h0',
                                         dim: 500,
                                         irange: .05,
                                         init_bias: -2,
                                         #max_col_norm: 1.9365,
                                     },
                                     !obj:pylearn2.models.mlp.Linear {
                                         #max_col_norm: 1.9365,
                                         # init_bias: !obj:pylearn2.models.dbm.init_sigmoid_bias_from_marginals { dataset: *train},
                                         layer_name: 'fc_subnet_y',
                                         irange: .05,
                                         dim: 3072
                                     },
                                     !obj:pylearn2.models.mlp.SpaceConverter {
                                         layer_name: 'fc_subnet_converter',
                                         output_space: !obj:pylearn2.space.Conv2DSpace {
                                        shape: [32, 32],
                                        num_channels: 3,
                                        axes: ['c', 0, 1, 'b'],
                                    }},
                                 ] # end fc_subnet_layers
                             } # end fc_subnet
                         ] # end composite components
                     }, # end composite
                     !obj:galatea.adversarial.Sum {
                        layer_name: "summer"
                     }
                    ],
            nvis: 100,
        }""")

import numpy as np
from pylearn2.utils import sharedX
Z = sharedX(np.zeros((128, 100)))
X = mlp.fprop(Z)
assert X.tag.test_value.shape == (3, 32, 32, 128)
