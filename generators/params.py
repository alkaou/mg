# hyperparameters
vocab_size = 21578
batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 1024  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100

Epochs = max_iters // eval_interval

learning_rate = 1e-6
eval_iters = 200
n_embd = 768
n_head = 12
n_layer = 12
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