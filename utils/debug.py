import signal

def _debug(signal_number, interrupted_frame):
    import pdb
    import sys
    signal.signal(signal.SIGTSTP, signal.SIG_DFL)
    try:
        pdb.set_trace()
    finally:
        signal.signal(signal.SIGTSTP, _debug)


def setdebug():
    signal.signal(signal.SIGTSTP, _debug)

