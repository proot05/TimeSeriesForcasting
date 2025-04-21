import torch.nn.functional as F
from torch import nn
import torch

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.25, gamma=0.25, high_dilation=2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.high_dilation = high_dilation

        # Define the same kernel for both low and high derivatives
        kernel = torch.tensor([[-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0]], dtype=torch.float32).view(1, 1, 7)
        self.register_buffer("derivative_kernel_low", kernel)
        self.register_buffer("derivative_kernel_high", kernel)  # same values, but applied with dilation

    def conv_derivative_low(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, T) → (B, T, 1)
        x = x.transpose(1, 2)    # (B, F, T)
        kernel = self.derivative_kernel_low.to(x.device)
        dx = F.conv1d(x, kernel, padding=3)  # padding=2 for 5-element kernel
        dx = dx.transpose(1, 2)  # (B, T, F)
        return dx.squeeze(-1) if dx.shape[-1] == 1 else dx

    def conv_derivative_high(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)  # (B, T) → (B, T, 1)
        x = x.transpose(1, 2)  # (B, F, T)
        kernel = self.derivative_kernel_high.to(x.device)  # shape: (1, 1, 5)
        # Compute padding for dilated convolution
        dilation = self.high_dilation
        effective_kernel_size = (kernel.shape[-1] - 1) * dilation + 1
        padding = effective_kernel_size // 2
        dx = F.conv1d(x, kernel, padding=padding, dilation=dilation)
        dx = dx.transpose(1, 2)  # (B, T, F)
        return dx.squeeze(-1) if dx.shape[-1] == 1 else dx

    def log_weighted_spectral_loss(self, y_true, y_pred, eps=1e-6):
        true_fft = torch.fft.fft(y_true, dim=-1)
        pred_fft = torch.fft.fft(y_pred, dim=-1)

        true_mag = torch.log(torch.abs(true_fft) + eps)
        pred_mag = torch.log(torch.abs(pred_fft) + eps)

        N = y_true.shape[-1]
        freqs = torch.fft.fftfreq(N, d=1.0).to(y_true.device)
        weights = (torch.abs(freqs) ** 2).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, N)

        return F.mse_loss(weights * pred_mag, weights * true_mag)

    def forward(self, y_pred, y_true):
        # Base MSE
        mse = F.mse_loss(y_pred, y_true)

        # Spectral (Fourier-domain log-weighted MSE)
        # spectral = self.log_weighted_spectral_loss(y_true, y_pred)

        # Convolutional Derivative Loss
        dy_true_low = self.conv_derivative_low(y_true)
        dy_pred_low = self.conv_derivative_low(y_pred)
        derivative_low = F.mse_loss(dy_pred_low, dy_true_low)

        # Convolutional Derivative Loss
        dy_true_high = self.conv_derivative_high(y_true)
        dy_pred_high = self.conv_derivative_high(y_pred)
        derivative_high = F.mse_loss(dy_pred_high, dy_true_high)

        # Final weighted loss
        return self.alpha * mse + self.beta * derivative_low + self.gamma * derivative_high
