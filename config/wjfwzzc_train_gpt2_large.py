# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'lab.vc.nanoGPT.owt'
wandb_run_name='gpt2-772M'

# these make the total batch size be ~0.5M
# 16 batch size * 1024 block size * 4 gradaccum * 8 GPUs = 524,288
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 4

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# model
n_layer = 36
n_head = 20
n_embd = 1280

# adamw optimizer
learning_rate = 2.5e-4 # max learning rate
min_lr = 2.5e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
