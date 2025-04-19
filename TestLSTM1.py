from data.preprocess import TimeSeriesPreprocessor, MembraneDataLoader
from tests.predict import predict
import matplotlib.pyplot as plt
import torch
from models.lstm1 import MyLSTM
from tqdm import tqdm
from os.path import exists as ose
import shutil
import os
import numpy as np
from scipy.interpolate import interp1d

device = torch.device('cuda')

input_dir = 'train/train_output/LSTM1'
output_dir = 'tests/test_output/LSTM1'

if ose(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

state = TimeSeriesPreprocessor.load_state(input_dir + '/checkpoints' + '/LSTM1_normalizer.pkl')
new_dt = state['dt_new']
x_normalizer = state['normalizer']
seq_len = state['seq_len']

checkpoint = torch.load(input_dir + '/checkpoints' + '/checkpoint_50.pt', map_location=device)

model = MyLSTM(InFeatures=1,
                OutFeatures=1,
                num_layers=1,
                HiddenDim=256,
                FeedForwardDim=512,
                nonlinearity = 'elu')

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# load data from desired location in dataset folder
loader = MembraneDataLoader(date="4-17-25", frequency=0.67, number=1)
mem_time, mem_data = loader.load_data()

buffer = 2000

mem_time_test = mem_time[-(seq_len+buffer):]
mem_data_test = mem_data[-(seq_len+buffer):]

rnn_delay = 12.6

pred_id = [None] * seq_len
pred_id_time = [None] * seq_len

for i in tqdm(range(int((len(mem_time_test) - seq_len)))):

    pred, dt = predict(model, mem_time_test[i:i + seq_len], mem_data_test[i:i + seq_len], new_dt, rnn_delay,
                           seq_len, device, x_normalizer)
    pred_id.append(pred)

    pred_id_time.append(mem_time_test[i + seq_len - 1] + dt)

plt.figure(figsize=(100, 40))
lw = 3
plt.plot(mem_time_test, mem_data_test, color='C1', linestyle='solid', linewidth=lw, alpha=1,
            label='ind')
plt.plot(pred_id_time, pred_id, color='C0', linestyle='dashed', linewidth=lw, alpha=1,
        label='pred ind')
plt.legend(loc='lower right', frameon=True)
plt.xlabel('Time (s)')
plt.ylabel('Membrane Index')
# plt.xlim([pred_id_time[-1]-50,pred_id_time[-1]])
plt.savefig(output_dir + '/prediction.png', bbox_inches='tight')
plt.close()

mem_time_eval = mem_time_test[seq_len+10:]
mem_data_eval = mem_data_test[seq_len+10:]

pred_time_eval = pred_id_time[seq_len+10:]
pred_data_eval = pred_id[seq_len+10:]

# Define a common time base
common_time = np.linspace(
    max(min(mem_time_eval), min(pred_time_eval)),
    min(max(mem_time_eval), max(pred_time_eval)),
    num=max(len(mem_time_eval), len(pred_time_eval))
)

# Interpolate both time series
interp_mem_data = interp1d(mem_time_eval, mem_data_eval, kind='linear', fill_value="extrapolate")
interp_preds_data = interp1d(pred_time_eval, pred_data_eval, kind='linear', fill_value="extrapolate")

mem_data_interp = interp_mem_data(common_time)
preds_data_interp = interp_preds_data(common_time)

# 3. Compute error as a function of time
error = np.abs(mem_data_interp - preds_data_interp)

# 4. Print average error (mean absolute error)
average_error = np.mean(error)
print(f"Average Absolute Error: {average_error:.4f}")

# 5. Plot error over time
plt.figure(figsize=(10, 5))
plt.plot(common_time, error, label=f' Average Absolute Error = {average_error:.2f}')
plt.xlabel("Time")
plt.ylabel("Error")
plt.title("Time-varying Error Between Interpolated Time Series")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir + '/prediction_error.png', bbox_inches='tight')
plt.close()
