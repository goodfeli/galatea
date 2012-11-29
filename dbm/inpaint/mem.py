batch_size = 1250
mf_iter = 100
state_size = (784 + 500 + 1000)
total_state = batch_size * mf_iter * state_size
num_parameters = 784 + 500 + 1000 + 784 * 500 + 500 * 1000
num_param_buffers = 4 # param, grad, orig, prev direction
total_param_buffer = num_parameters * num_param_buffers

num_floats = total_state + total_param_buffer
num_bytes = num_floats*4
print float(num_bytes) / float(1024 ** 3)

