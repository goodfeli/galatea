
def matricize(y):
    assert y.ndim == 1
    return y.reshape(y.size, 1)
