import torch

def range_push(msg):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(msg)

def range_pop():
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(msg)