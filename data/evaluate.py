import torch

def percent_variance_explained(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """
    pred, gt: shape (T,) or (T,C), same dtype (float).
    Returns percentage of variance in gt explained by pred.
    """
    # flatten to (T*C,)
    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    ss_res = torch.sum((g - p)**2)
    ss_tot = torch.sum((g - torch.mean(g))**2)
    r2 = 1 - ss_res/ss_tot
    return (r2 * 100).item()


