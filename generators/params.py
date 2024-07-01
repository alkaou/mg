# hyperparameters
vocab_size = 68610
batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 1024  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100

Epochs = max_iters // eval_interval

learning_rate = 1e-6
eval_iters = 200
n_embd = 192
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

# hyperparameters pour teste
# vocab_size = 5000
# batch_size = 16  # how many independent sequences will we process in parallel?
# block_size = 32  # what is the maximum context length for predictions?
# max_iters = 5000
# eval_interval = 100

# Epochs = max_iters // eval_interval

# learning_rate = 1e-6
# eval_iters = 200
# n_embd = 12
# n_head = 2
# n_layer = 2
# dropout = 0.0
# ------------