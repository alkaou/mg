# hyperparameters
vocab_size = 20000
batch_size = 16  # how many independent sequences will we process in parallel?
block_size = 32  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100

Epochs = max_iters // eval_interval

learning_rate = 1e-6
eval_iters = 200
n_embd = 1064
n_head = 10
n_layer = 6
dropout = 0.0
# ------------