#! /usr/bin/env python

import sys, os

os.environ['THEANO_FLAGS'] = 'device=gpu,floatX=float32'

# Fake running jtrain.py:
#sys.argv.append('fish_model_ccn.yaml')
class Container(): pass

args = Container
args.config = 'fish_model_ccn.yaml'
args.name = 'junk'
args.diary = False
args.quick = True
args.timestamp = False
args.verbose_logging = False
args.debug = False

from pylearn2.scripts.jtrain import *
main2(args)
