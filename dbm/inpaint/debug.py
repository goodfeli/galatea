from pylearn2.utils import serial
from pylearn2.config import yaml_parse

model = yaml_parse.load("""
 !obj:galatea.dbm.inpaint.super_dbm.SuperDBM {
              batch_size : 10,
              niter: 6, #note: since we have to backprop through the whole thing, this does
                         #increase the memory usage
              visible_layer: !obj:galatea.dbm.inpaint.super_dbm.GaussianConvolutionalVisLayer {
                rows: 32,
                cols: 32,
                channels: 3,
                init_beta: 3.7, # this is the average marginal beta
                min_beta: 3.7, # this is the average marginal beta
                tie_beta: 'locations',
                init_mu: 0.,
                tie_mu: 'locations'
              },
              hidden_layers: [
                !obj:galatea.dbm.inpaint.super_dbm.ConvMaxPool {
                        mirror_weights: 1,
                        output_channels: 64,
                        border_mode : 'full',
                        kernel_rows: 7,
                        kernel_cols: 7,
                        pool_rows: 2,
                        pool_cols: 2,
                        irange: .05,
                        layer_name: 'h0',
                        init_bias: -3.
               }
              ]
    }
        """)

model.dataset_yaml_src = """!obj:galatea.datasets.zca_dataset.ZCA_Dataset {
        preprocessed_dataset: !pkl: "/data/lisa/data/cifar10/pylearn2_gcn_whitened/train.pkl",
        preprocessor: !pkl: "/data/lisa/data/cifar10/pylearn2_gcn_whitened/preprocessor.pkl"
    }"""

serial.save("debug.pkl",model)
