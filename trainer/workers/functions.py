import torch
import torch.distributed as dist


def _differentiable_all_reduce(
    tensor: torch.Tensor, process_group: dist.ProcessGroup
) -> torch.Tensor:
    # this does requires_grad = False
    # This is importance to do NCCL primitives, as PyTorch autograd can't track NCCL communication
    # So we do this dummy operation (+ followed by -) after we do all reduce on the detached tensor
    # Finally the tensor is requires_grad = True
    detached_tensor = tensor.detach()
    dist.all_reduce(detached_tensor, op=dist.ReduceOp.SUM, group=process_group)
    return tensor + detached_tensor - tensor.detach()


def _compute_logsumexp(
    logits: torch.Tensor, process_group: dist.ProcessGroup, chunk_size: int = 1024
) -> torch.Tensor:
    logsumexps: list[torch.Tensor] = []
    for start in range(0, logits.shape[1], chunk_size):
        logsumexp = torch.logsumexp(logits[:, start : start + chunk_size], -1)
        logsumexps.append(logsumexp)
    logsumexp_concat = torch.cat(logsumexps, -1)

    logsumexps = [
        torch.zeros_like(logsumexp_concat)
        for _ in range(dist.get_world_size(process_group))
    ]

    dist.all_gather(logsumexps, logsumexp_concat, group=process_group)
    logsumexps[dist.get_rank(process_group)] = logsumexp
    logsumexps = torch.cat([logsumexp.unsqueeze(-1) for logsumexp in logsumexps], -1)
    return torch.logsumexp(logsumexps, -1)
