from pylearn2.train_extensions import TrainExtension

class OnMonitorError(Exception):
    """
    Raised by ErrorOnMonitor when on_monitor is called.
    """

class ErrorOnMonitor(TrainExtension):
    def on_monitor(self, *args, **kwargs):
        raise OnMonitorError()
