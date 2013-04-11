from model_dbm_inpainting import DBMinpainting
from natSGD import natSGD
from mainLoop import MainLoop
from dataMNIST_standard import DataMNIST
import numpy

def jobman(state, channel):
    rng = numpy.random.RandomState(state['seed'])
    model = DBMinpainting(state)
    data = DataMNIST(state['path'], state['mbs'], state['bs'], rng,
                     same_batch=state['samebatch'],
                     callback=model.callback)

    algo = natSGD(model, state, data)
    main = MainLoop(data, model, algo, state, channel)
    main.main()

if __name__=='__main__':
    state = {}

    state['samebatch'] = 1
    #state['path'] = '/scratch/pascanur/data/'
    #state['path'] = '/data/lisa/data/faces/TFD/'
    state['path'] = '/RQexec/pascanur/data/mnist.npz'

    state['mbs'] = 128*2
    state['bs']  = 128*2
    state['ebs'] = 128*2
    state['cbs'] = 128*2

    state['loopIters'] = 6000
    state['timeStop'] = 32*60
    state['minerr'] = 1e-5

    state['lr'] = .2
    state['lr_adapt'] = 0

    state['damp'] = 5.
    state['adapt'] = 1.
    state['mindamp'] = .5
    state['damp_ratio'] =5./4.
    state['minerr'] = 1e-5

    state['seed'] = 123


    state['profile'] = 0
    state['minresQLP'] = 1
    state['mrtol'] = 1e-4
    state['miters'] = 50
    state['trancond'] = 1e-4


    state['trainFreq'] = 1
    state['validFreq'] = 2000
    state['saveFreq'] = 20

    state['prefix'] = 'conv_'
    state['overwrite'] = 1
    jobman(state, None)

