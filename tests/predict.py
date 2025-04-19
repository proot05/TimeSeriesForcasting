from scipy.interpolate import interp1d
import numpy as np
import torch
import math

def predict(model, indtime_history, ind_history, dt_new, rnn_delay, seq_len, device, x_normalizer=None):

    # interpolation
    interp_func = interp1d(indtime_history, ind_history, kind='linear', fill_value='extrapolate')
    ind_times_new = indtime_history[-1]-np.flip(np.arange(seq_len))*dt_new
    ind_acquired_new = interp_func(ind_times_new)

    history = ind_acquired_new[..., None][-seq_len:]

    deltat = rnn_delay * dt_new
    matched_idx, pred = predict_data(model, device, history, deltat, dt_new=dt_new, seq_len=seq_len, x_normalizer=x_normalizer)

    return matched_idx, deltat


def predict_data(model, device, history, deltat, dt_new, seq_len, x_normalizer=None):

    with torch.no_grad():
        # calculate inference length
        steps = math.ceil(deltat / dt_new)
        ratio = np.remainder(deltat, dt_new) / dt_new
        predicted_data = x_normalizer.normalize(torch.tensor(history[-seq_len:], dtype=torch.float32))  # <seq_len, 1>

        for i in range(steps):
            output = model(predicted_data.unsqueeze(0).to(device))  # <B,L,C>
            predicted_data = torch.cat((predicted_data, output[0, -1].unsqueeze(-1).detach().cpu()), dim=0)
    predicted_data_denorm = x_normalizer.denormalize(predicted_data).numpy()

    pred_idx = int((1 - ratio) * predicted_data_denorm[-2] + ratio * predicted_data_denorm[-1])
    # pred_idx = int(predicted_data_denorm[-1])

    return pred_idx, predicted_data_denorm
