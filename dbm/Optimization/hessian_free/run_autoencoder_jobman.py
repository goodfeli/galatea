# jobman cmdline run_autoencoder_jobman.main

from jobman import DD, expand
from params_default_hf import default_config as config
import sys
import run_autoencoder
from monitor import exp_monitor

def set_config(conf, args):
    for key in args:
        if key != 'jobman':
            v = args[key]
            if isinstance(v, DD):
                set_config(conf[key], v)
            else:
                if conf.has_key(key):
                    conf[key] = convert_from_string(v)
                else:
                    #import ipdb; ipdb.set_trace()
                    raise KeyError(key)

def convert_from_string(x):
        """
        Convert a string that may represent a Python item to its proper data type.
        It consists in running `eval` on x, and if an error occurs, returning the
        string itself.
        """
        try:
            return eval(x, {}, {})
        except Exception:
            return x
                        

def main(state, channel=None):
    
    exp_monitor.init_jobman_channel(channel, state)

    # Copy state into config.
    set_config(config, state)

    # Also copy back from config into state.
    for key in config:
        setattr(state, key, config[key])

    params = state.copy()

    if params.has_key('jobman'):
        # in case of running by using jobman cmdline ....
        params.pop('jobman')
    
    run_autoencoder.jobman_main(params)

    if channel != None:
        return channel.COMPLETE
    
if __name__ == '__main__':
    args = {}
    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            args[k] = v
    except:
        print 'args must be like a=X b.c=X'
        exit(1)
    
    state = expand(args)
    
    sys.exit(main(state))
