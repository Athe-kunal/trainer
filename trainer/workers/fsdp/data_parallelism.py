from typing import Type
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def _param_init_fn(module: nn.Module):
    # Reserve the slots on GPU, but don’t copy real weights from CPU into them yet.
    # Construct the model on the meta device (parameters have shapes but no storage anywhere)
    # Then later materialize weights directly onto GPU or shard-load them
    module.to_empty(device=torch.cuda.current_device(), recurse=False)


def prepare_dp_model(
    model: nn.Module,
    dtype: str,
    sync_module_states: bool,
    device_mesh: dist.DeviceMesh,
) -> FSDP:

    def get_module_cls_from_name(name: str) -> Type[nn.Module] | None:
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__
        raise ValueError(f"Module {name} not found in model")

    transformer_layer_cls = [
        # the no split modules will be something like LlamaDecoderLayer
        get_module_cls_from_name(name)
        for name in model._no_split_modules
    ]

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls,
    )
    dtype: torch.dtype = getattr(torch, dtype)
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        buffer_dtype=dtype,
        reduce_dtype=dtype,
    )
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=mixed_precision,
        param_init_fn=_param_init_fn,
        sync_module_states=sync_module_states,
        device_mesh=device_mesh,
        device_id=torch.cuda.current_device(),
    )
