import torch


def highpass_tensor(x: torch.Tensor, fs: float, cutoff: float) -> torch.Tensor:
    N = x.shape[-1]
    # Compute real FFT
    Xf = torch.fft.rfft(x)
    # Frequency bins
    freqs = torch.fft.rfftfreq(n=N, d=1/fs, device=x.device)
    # Zero out anything below cutoff
    mask = freqs >= cutoff
    Xf = Xf * mask
    # Inverse FFT back to time domain
    x_hp = torch.fft.irfft(Xf, n=N)
    return x_hp

def high_freq_snr(pred: torch.Tensor, gt: torch.Tensor, fs: float, cutoff: float) -> float:
    """
    Compute SNR of the high‑pass component above `cutoff` Hz,
    for torch.Tensor inputs.

    pred, gt : torch.Tensor of shape (..., N)
    fs        : sampling rate in Hz
    cutoff    : high‑pass cutoff frequency in Hz
    """
    # flatten both signals
    p = pred.contiguous().view(-1)
    g = gt.contiguous().view(-1)

    # high‑pass filter
    y_hf = highpass_tensor(g,  fs, cutoff)
    yhat_hf = highpass_tensor(p, fs, cutoff)

    # signal and noise powers
    P_signal = torch.mean(y_hf**2)
    P_noise = torch.mean((y_hf - yhat_hf)**2)

    # SNR in dB
    snr_db = 10 * torch.log10(P_signal / (P_noise + 1e-12))
    return snr_db.item()


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
