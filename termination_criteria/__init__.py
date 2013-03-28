
class Throwaway(object):

    def continue_learning(self, model):
        monitor = model.monitor
        epochs_seen = monitor.get_epochs_seen()
        valid_err = monitor.channels['valid_err'].val_record[-1]
        print 'Throwaway.continue_learning:'
        print '\tepochs_seen: ',epochs_seen
        print '\tvalid_err: ',valid_err
        print '\t(Yes, I called [-1])'
        if epochs_seen > 10 and epochs_seen <= 20:
            return valid_err <= .9
        if epochs_seen > 20 and epochs_seen <= 30:
            return valid_err < .8
        if epochs_seen > 30 and epochs_seen <= 40:
            return valid_err < .7
        if epochs_seen > 40 and epochs_seen <= 50:
            return valid_err < .6
        if epochs_seen > 50 and epochs_seen <= 60:
            return valid_err < .5
        return True
