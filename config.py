
# training config
SEED = 0
LR = 1e-3
BATCH_SIZE = 4
EPOCHS = 30
PRINT_FREQ = 1
VAL_FREQ = 1

# inference config
SAMPLE_SIZE = 16
LOAD_PATH = "ckpts/best.pt"
TRANSFER_IDX = 0

# model config
nc_img = 3
nc_cond = 1
n_filter = 32
n_key = 8
dim_key = 4
dim_val = 4

# Dataset config
DATA_ROOT = "dataset/cars"
