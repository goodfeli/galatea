import base64
import sys

def moo():
    '''
    A cute function that moos.
    '''
    print sys.argv[0]
    # moo from base64
    print base64.b64decode('CiAgICAgICAgIChfXykKICAg\
                            ICAgICAgKG9vKQogICAvLS0tL\
                            S0tXC8KICAvIHwgICAgfHwKICo\
                            gIC9cLS0tL1wgCiAgICB+fiAgI\
                            H5+Ckl0IHdvcmtzIQo=')

# Moo!
moo()
