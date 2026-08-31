import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
x = torch.ones(1, device=f"cuda:{local_rank}")
dist.all_reduce(x)
print(f"rank {rank} (local_rank {local_rank}) all_reduce result: {x.item()}", flush=True)
dist.destroy_process_group()
