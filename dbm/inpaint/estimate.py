input_rows = 32
input_cols = 32
input_ch   = 3
kernel_rows = 7
kernel_cols = 7
output_ch = 30

data_size = input_rows * input_cols * input_ch
mask_size = input_rows * input_cols
beta_size = input_rows * input_cols * input_ch
mu_size   = input_rows * input_cols * input_ch
b_size = output_ch
kernel_size = output_ch * input_ch * kernel_rows * kernel_cols

h_rows = input_rows - kernel_rows + 1
h_cols = input_cols - kernel_cols + 1

h_size = h_rows * h_cols * output_ch


