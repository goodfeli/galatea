
def matricize(y):
    assert y.ndim == 1
    return y.reshape(y.size, 1)

def get_prepro(dataset):
    return dataset.preprocessor
