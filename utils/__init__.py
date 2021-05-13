import torch


def round_preserve_sum(t: torch.Tensor) -> torch.LongTensor:
    """
    Rounds tensor while preserving their rounded sum.
    Algorithm:
        1. Floor the tensor t
        2. Order indices by their remainder values
        3. Increment the values with k largest remainders,
            where k is the amount that we need to increase the sum to reach our target value
    Do not expect this to maintain gradients
    """
    t_rounded = t.floor().to(torch.long)
    t_remainders = t - t_rounded
    t_ks = t.sum(dim=1).round() - t_rounded.sum(dim=1)
    while torch.count_nonzero(t_ks) > 0:
        for batch_idx, incr_idx in enumerate(t_remainders.argmax(dim=1)):
            if t_ks[batch_idx] > 0:
                t_rounded[batch_idx, incr_idx] += 1
                t_remainders[batch_idx, incr_idx] -= 1
                t_ks[batch_idx] -= 1
    return t_rounded
