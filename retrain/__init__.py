import numpy as np

def re_initialize(mlp, re_initialize):

    rng = np.random.RandomState([1, 2, 3])

    if not hasattr(mlp, 'monitor_stack'):
        mlp.monitor_stack = [mlp.monitor]
    else:
        mlp.monitor_stack.append(mlp.monitor)
    del mlp.monitor

    for idx in re_initialize:
        layer = mlp.layers[idx]
        for param in layer.get_params():
            if param.ndim == 2:
                value = param.get_value()
                value = rng.uniform(-layer.irange, layer.irange, value.shape)
                param.set_value(value.astype(param.dtype))
            else:
                assert param.ndim == 1
                value = param.get_value()
                value *= 0
                value += layer.bias_hid
                param.set_value(value.astype(param.dtype))

    return mlp
