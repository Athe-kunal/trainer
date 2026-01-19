from torch.nn.parameter import Parameter


import os
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.distributed import device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel


import warnings

warnings.filterwarnings("ignore")


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

        with torch.no_grad():
            self.net1_weight = nn.Parameter(torch.arange(101.0, 201.0))

    def forward(self, x):
        return self.relu(self.net1(x))


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = "nccl" if dist.is_nccl_available() else "gloo"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    if device == "cuda":
        torch.cuda.set_device(local_rank)

    # row major order
    # device_mesh_ = device_mesh.init_device_mesh(
    #     device, mesh_dim_names=("dp", "tp"), mesh_shape=(2, 2)
    # )
    # tp_mesh = device_mesh_["tp"]
    # print(f"Rank {dist.get_rank()}, {tp_mesh=} ")

    model = ToyModel()

    mesh = device_mesh.init_device_mesh(device, (2, 2), mesh_dim_names=["DP", "FSDP"])

    # model = FSDP(
    #     model,
    #     device_mesh=mesh,
    #     sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    # )
    # model = fully_shard(model, mesh=mesh["FSDP"])  # Here the weights would be DTensor
    print(
        f'Global rank: {dist.get_rank()}, dp_rank: {mesh["DP"].get_local_rank()}, fsdp_rank:{mesh["FSDP"].get_local_rank()}, Weights shapes: {[p.shape for p in model.parameters()]}'
    )

    # Tensor Parallel

    mesh = device_mesh.init_device_mesh(device, (2, 2), mesh_dim_names=["FSDP", "TP"])
    model = parallelize_module(
        model,
        mesh["TP"],
        {
            "net1": ColwiseParallel(),
        },
    )

    model = fully_shard(
        model, mesh=mesh["FSDP"], reshard_after_forward=True
    )  # Here the weights would be DTensor
    print(
        f'Global rank: {dist.get_rank()}, fsdp_rank: {mesh["FSDP"].get_local_rank()}, tp_rank:{mesh["TP"].get_local_rank()}, Weights array: {list(model.parameters())}'
    )
    # Whether to reshard after all gather
    full_tensor = [param.full_tensor() for param in model.parameters()]
    if dist.get_rank() == 0:
        print(full_tensor)
    dist.destroy_process_group()
