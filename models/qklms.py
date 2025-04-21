import torch


# === Rational Quadratic Kernel for QKLMS ===
class RationalQuadraticKernel(torch.nn.Module):
    def __init__(self, sigma=1.0, alpha=1.0):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha

    def forward(self, x, y):
        diff = x - y
        dist2 = torch.sum(diff**2, dim=1)
        return torch.pow(1 + dist2 / (2 * self.alpha * self.sigma**2), -self.alpha)


# === QKLMS Model (Correcting Historical LSTM Errors) ===
class QKLMS(torch.nn.Module):
    def __init__(self, input_dim, sigma=1.0, alpha=1.0, gain_lambda=3.0, memory=500, top_k=3):
        super().__init__()
        self.kernel = RationalQuadraticKernel(sigma, alpha)
        self.gain_lambda = gain_lambda
        self.memory = memory
        self.top_k = top_k
        self.input_dim = input_dim
        self.register_buffer("X", torch.empty((0, input_dim)))
        self.alpha = []
        self.errors = []  # This will store the historical prediction errors

    def predict(self, x):
        if self.X.shape[0] == 0:
            return torch.tensor(0.0, device=x.device)  # Initial prediction if no data
        k_vals = self.kernel(x.expand(self.X.size(0), -1), self.X)
        k = min(self.top_k, k_vals.shape[0])
        topk_vals, topk_idx = torch.topk(k_vals, k=k, largest=True)
        pred = torch.sum(topk_vals * torch.tensor([self.alpha[i] for i in topk_idx.tolist()], device=x.device))
        confidence = torch.max(topk_vals)
        return pred * (1 + self.gain_lambda * confidence)

    def update(self, x, target, eta):
        # Predict based on past errors and input data
        y_hat = self.predict(x)
        error = target - y_hat  # The error between prediction and true value
        self.X = torch.cat([self.X, x.unsqueeze(0)], dim=0)
        self.alpha.append((eta * error).item())
        self.errors.append(error.item())  # Store the current error to learn from it
        if self.X.shape[0] > self.memory:
            self.X = self.X[1:]
            self.alpha.pop(0)
            self.errors.pop(0)
        return y_hat, error
