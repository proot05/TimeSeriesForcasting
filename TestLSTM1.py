from data.preprocess import TimeSeriesPreprocessor, MembraneDataLoader
from tests.predict import rnn_predict
import matplotlib.pyplot as plt
import torch
from models.lstm1 import MyLSTM
from tqdm import tqdm
from os.path import exists as ose
import shutil
import os

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

for i in tqdm(range(len(mem_time_test) - seq_len)):
    pred, dt = rnn_predict(model, mem_time_test[i:i + seq_len], mem_data_test[i:i + seq_len], new_dt, rnn_delay,
                           seq_len, device, x_normalizer)
    pred_id.append(pred)
    pred_id_time.append(mem_time_test[i + seq_len] + dt)

plt.figure(figsize=(100, 40))
lw = 3
plt.plot(mem_time_test, mem_data_test, color='C1', linestyle='solid', linewidth=lw, alpha=1,
            label='ind')
plt.plot(pred_id_time, pred_id, color='C0', linestyle='dashed', linewidth=lw, alpha=1,
        label='pred ind')
plt.legend(loc='lower right', frameon=True)
plt.xlabel('Time (s)')
plt.ylabel('Membrane Index')
plt.xlim([pred_id_time[-1]-50,pred_id_time[-1]])
plt.savefig(output_dir + '/prediction.png', bbox_inches='tight')
plt.close()
