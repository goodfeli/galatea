#! /usr/bin/env python

import sys, os, pdb

os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'

from pylearn2.scripts.jtrain import *
try:
    main('--quick --nogrm fish_model_ccn.yaml.quick'.split())
except:
    pdb.set_trace()
