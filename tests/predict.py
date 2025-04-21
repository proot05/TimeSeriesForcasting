from scipy.interpolate import interp1d
import numpy as np
import torch
import math


@torch.no_grad()
def predict(model, indtime_history, ind_history, dt_new, rnn_delay, seq_len, device, x_normalizer=None):

    # interpolation
    interp_func = interp1d(indtime_history, ind_history, kind='linear', fill_value='extrapolate')
    ind_times_new = indtime_history[-1]-np.flip(np.arange(seq_len))*dt_new
    ind_acquired_new = interp_func(ind_times_new)

    history = ind_acquired_new[..., None][-seq_len:]

    # calculate inference length
    steps = math.ceil(rnn_delay / dt_new)
    ratio = (rnn_delay - dt_new*(steps - 1)) / dt_new

    # print("loops: ", steps)
    # print("dt: ", dt_new)

    init = torch.tensor(history[-seq_len:], dtype=torch.float32)
    init_n = x_normalizer.normalize(init).to(device)  # <seq_len, 1>
    total = seq_len + steps
    predicted_data = torch.empty((total, 1), device=device)
    predicted_data[:seq_len] = init_n

    hidden = None
    for i in range(steps):
        window = predicted_data[i:i+seq_len].unsqueeze(0)
        output, hidden = model(window, hidden)  # <B,L,C>
        predicted_data[seq_len + i] = output[0, -1]

    predicted_data_denorm = x_normalizer.denormalize(predicted_data).detach().cpu().numpy()

    if rnn_delay == 0:
        return int(predicted_data_denorm[-1])

    pred_idx = int((1 - ratio) * predicted_data_denorm[-2] + ratio * predicted_data_denorm[-1])

    return pred_idx
