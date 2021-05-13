import torch


def softclamp(x: torch.Tensor, maximums: torch.Tensor, a=1.0, threshold=20.0):
    """
    Applies the element-wise soft-clamp function:

        Softplus(x_i, a_i) * (1 - Sigmoid(a_i * (x_i - maximums_i/2))) +
            (Softplus(x_i - maximums_i, -a_i) + maximums_i) * Sigmoid(a_i * (x_i - maximums_i/2))

    x, a, b and threshold must all be broadcastable.
    """
    upper_mix = torch.sigmoid(a * (x - (maximums / 2)))
    lower_mix = 1 - upper_mix

    # Figure out which elements go over the threshold.
    lower_x = a * x
    upper_x = a * (maximums - x)
    lower_x_mask = lower_x > threshold
    upper_x_mask = upper_x > threshold

    # we'll sub their values back in later
    lower_x_overflows = lower_x.masked_select(lower_x_mask)
    upper_x_overflows = upper_x.masked_select(upper_x_mask)

    # next compute the softpluses
    lower_x_exponent = lower_x.masked_fill(lower_x_mask, threshold)
    upper_x_exponent = upper_x.masked_fill(upper_x_mask, threshold)
    lower = (1 / a) * torch.log(1 + torch.exp(lower_x_exponent))
    upper = - (1 / a) * torch.log(1 + torch.exp(upper_x_exponent)) + maximums
    lower = lower.masked_scatter(lower_x_mask, lower_x_overflows)
    upper = upper.masked_scatter(upper_x_mask, upper_x_overflows)

    # Mix it all together
    result = lower * lower_mix + upper * upper_mix
    return result
