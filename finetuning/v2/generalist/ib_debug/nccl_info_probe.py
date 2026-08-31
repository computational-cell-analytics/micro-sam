import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
torch.cuda.set_device(0)
dist.init_process_group("nccl")
x = torch.ones(1, device="cuda:0")
dist.all_reduce(x)
dist.destroy_process_group()
