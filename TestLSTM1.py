from data.preprocess import TimeSeriesPreprocessor, MembraneDataLoader
from testfuncs.predict import predict
import matplotlib.pyplot as plt
import torch
from models.lstm1 import MyLSTM
from tqdm import tqdm
from os.path import exists as ose
import shutil
import os
import numpy as np
from scipy.interpolate import interp1d
from data.evaluate import percent_variance_explained, smape
import pylab

device = torch.device('cuda')

multp = 6
params = {'legend.fontsize': 25*multp, 'axes.labelsize': 25*multp, 'axes.titlesize': 25*multp, 'xtick.labelsize': 25*multp,
          'ytick.labelsize': 25*multp}
pylab.rcParams.update(params)

input_dir = 'train/train_output/LSTM1'
output_dir = 'testfuncs/test_output/LSTM1'

if ose(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

state = TimeSeriesPreprocessor.load_state(input_dir + '/checkpoints' + '/LSTM1_normalizer.pkl')
new_dt = state['dt_new']
x_normalizer = state['normalizer']
seq_len = state['seq_len']
train_size = 2000 # state['train_size']

checkpoint = torch.load(input_dir + '/checkpoints' + '/checkpoint_50.pt', map_location=device)

model = MyLSTM(InFeatures=1,
                OutFeatures=1,
                num_layers=1,
                HiddenDim=256,
                FeedForwardDim=512,
                nonlinearity = 'tanh')

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

# printing out the model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

model.eval()

# ~9 Hz sampled data (from the training data sample)
# loader = MembraneDataLoader(date="4-17-25", frequency=0.67, number=1)
# mem_time, mem_data = loader.load_data()
# buffer = 2000
# mem_time_test = mem_time[-(seq_len+buffer):]
# mem_data_test = mem_data[-(seq_len+buffer):]

# ~40 Hz sampled data (same as training but from a different sample than the training data)
loader = MembraneDataLoader(date="4-17-25", frequency=0.67, number=2)
mem_time, mem_data = loader.load_data()
mem_time_test = mem_time[:train_size]
mem_data_test = mem_data[:train_size]

rnn_delay = 0.25

pred_id = [None] * seq_len
pred_id_time = [None] * seq_len

for i in tqdm(range(int((len(mem_time_test) - seq_len)))):

    pred = predict(model, mem_time_test[i:i + seq_len], mem_data_test[i:i + seq_len], new_dt, rnn_delay,
                           seq_len, device, x_normalizer)
    pred_id.append(pred)

    pred_id_time.append(mem_time_test[i + seq_len - 1] + rnn_delay)

mem_time_eval = mem_time_test[seq_len:]
mem_data_eval = mem_data_test[seq_len:]

pred_time_eval = pred_id_time[seq_len:]
pred_data_eval = pred_id[seq_len:]

# Define a common time base
common_time = np.linspace(pred_time_eval[15], mem_time_eval[-1], num=max(len(mem_time_eval), len(pred_time_eval)))

# Interpolate both time series
interp_mem_data = interp1d(mem_time_eval, mem_data_eval, kind='linear', fill_value="extrapolate")
interp_preds_data = interp1d(pred_time_eval, pred_data_eval, kind='linear', fill_value="extrapolate")

mem_data_interp = interp_mem_data(common_time)
preds_data_interp = interp_preds_data(common_time)

#  Compute error as a function of time
error = mem_data_interp - preds_data_interp
# error = mem_data_interp - pred_data_eval

# Print average error (mean absolute error)
average_error = np.mean(np.abs(error))
print(f"MAE = {average_error:.4f}")

pct_var1 = percent_variance_explained(torch.tensor(preds_data_interp, dtype=torch.float32), torch.tensor(mem_data_interp, dtype=torch.float32))
print(f"R² % = {pct_var1:.2f}%")

pct_var2 = smape(torch.tensor(preds_data_interp, dtype=torch.float32), torch.tensor(mem_data_interp, dtype=torch.float32))
print(f"SMAPE = {pct_var2:.2f}%")

# Plotting
plt.figure(figsize=(30*multp, 24*multp))
lw = 3*multp
plt.plot(mem_time_test, mem_data_test, color='C1', linestyle='solid', linewidth=lw, alpha=1, label='Ground Truth')
plt.plot(pred_id_time, pred_id, color='C0', linestyle='dashed', linewidth=lw, alpha=1, label='Prediction')
plt.plot(pred_time_eval, error, color='C2', linestyle='solid', linewidth=lw, alpha=1, label='Error')
plt.legend(loc='lower left', frameon=True)
plt.xlabel('Time (s)')
plt.ylabel('Index')
plt.title(f'MAE = {average_error:.2f}, R² % = {pct_var1:.2f}%, SMAPE = {pct_var2:.2f}%')
# plt.xlim([pred_id_time[-1]-50, pred_id_time[-1]])
plt.savefig(output_dir + '/prediction.png', bbox_inches='tight')
plt.close()
