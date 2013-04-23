

def read_param(model, param):
    return getattr(model, param).get_value()
