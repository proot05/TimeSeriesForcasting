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


def smape(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error between prediction and ground truth.

    SMAPE = 100% * (1/N) * Σ |pred_i - gt_i| / ((|gt_i| + |pred_i|)/2 + eps)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values, any shape.
    gt : torch.Tensor
        Ground-truth values, same shape as pred.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        SMAPE in percent.
    """
    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)
    denom = (p.abs() + g.abs()) / 2.0 + eps
    smape_val = 100.0 * torch.mean((p - g).abs() / denom)
    return smape_val.item()
