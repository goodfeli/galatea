D = 32*32*3
N = 7000
N2 = 7000

num_chains = 100
batch_size = 100

chain_state_size = num_chains * (N + N2)
pos_state_size = batch_size * (D + N + N2)
parameters_size = D + D*N + N*3 + N*N2 + N2

total_size = chain_state_size + pos_state_size + parameters_size

#compensate for unavoidable temporaries
total_size = total_size * 3

size_in_bytes = total_size * 4

size_in_gb = float(size_in_bytes) / float(1024**3)

print size_in_gb
