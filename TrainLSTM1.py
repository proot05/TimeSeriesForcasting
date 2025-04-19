import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from os.path import exists as ose
import os
import shutil
import torch
from tqdm import tqdm
from data.preprocess import MyDatasetAutoregress, MembraneDataLoader, TimeSeriesPreprocessor
from train.train import test_rollout_autoregress, train_regular_transformer_autoregress
from models.lstm1 import MyLSTM
from models.loss import CombinedLoss

# load GPU
device = torch.device('cuda')

# load data from desired location in dataset folder
loader = MembraneDataLoader(date="4-17-25", frequency=0.67, number=1)
mem_time, mem_data = loader.load_data()

# set up the output directory
output_dir = 'train/train_output/LSTM1'

if ose(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(output_dir + '/data')
os.mkdir(output_dir + '/figs')
os.mkdir(output_dir + '/meshes')
os.mkdir(output_dir + '/checkpoints')
os.mkdir(output_dir + '/inference')

# parameters
train_size = 2000
seq_len = 400
batch_size = 16
forward_pred = 20

preprocessor = TimeSeriesPreprocessor(train_size=train_size, seq_len=seq_len, norm_method='ms', norm_dim=0, output_dir=output_dir)

# train and validation split
mem_time_train = mem_time[:train_size]
mem_data_train = mem_data[:train_size]

processed_train_inputs = preprocessor.process(mem_time_train, mem_data_train)

preprocessor.save_state(output_dir + '/checkpoints' + '/LSTM1_normalizer.pkl')

my_dataset = MyDatasetAutoregress(processed_train_inputs, seq_len, forward_pred)

train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

model = MyLSTM(InFeatures=1,
                OutFeatures=1,
                num_layers=1,
                HiddenDim=256,
                FeedForwardDim=512,
                nonlinearity = 'tanh')

model.to(device)

# printing out the model size
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# training parameters
epochs = 51
lr = 1e-4
save_interval = (epochs - 1)/5
gamma = 0.999

criterion = CombinedLoss(alpha=0.2, beta=0.2, gamma=0.6, high_dilation=10)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# start training
train_ins_error = []

for epoch in tqdm(range(epochs)):

    temp_error = train_regular_transformer_autoregress(model, device, train_loader, optimizer, criterion, forward_pred,
                                                       scheduler=scheduler,
                                                       x_normalizer=preprocessor.normalizer, epoch=epoch, output_dir=output_dir,
                                                       save_interval=save_interval)
    train_ins_error.append(temp_error.detach().cpu())

    if (epoch % save_interval == 0) and True:
        fig, ax = plt.subplots(figsize=(10, 8))
        lw = 1
        ax.plot(train_ins_error, color='C0', linestyle='solid', linewidth=lw, alpha=1, label='L_b_fit')
        ax.set_yscale('log')
        leg = ax.legend(loc='upper right', frameon=True)
        leg.get_frame().set_edgecolor('black')
        fig.savefig(output_dir + '/figs/train_error_epoch{:d}.png'.format(epoch), bbox_inches='tight')

        # save your model
        error1 = torch.stack(train_ins_error)
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': error1.detach().cpu(),
        }, output_dir + "/checkpoints/checkpoint_{:d}.pt".format(epoch))
        # print(['{:e}'.format(ele) for ele in error1])

        # rollout
        pred, gt = test_rollout_autoregress(model, device, processed_train_inputs, seq_len=seq_len, x_normalizer=preprocessor.normalizer)

        # plot out the prediction results
        pick_channel = 0
        # plot out the enery distribution
        fig, ax = plt.subplots(figsize=(10, 8))
        lw = 1

        ax.plot(gt.numpy()[:, pick_channel], color='C0', linestyle='solid', linewidth=lw, alpha=1, label='gt')
        ax.plot(pred.numpy()[:, pick_channel], color='C1', linestyle='solid', linewidth=lw, alpha=1,
                label='prediction')
        ax.plot(pred.numpy()[:, pick_channel] - gt.numpy()[:, pick_channel], color='C2', linestyle='solid',
                linewidth=lw, alpha=1, label='difference')
        leg = ax.legend(loc='lower right', frameon=True)
        leg.get_frame().set_edgecolor('black')
        ax.set_xlabel('time')
        ax.set_ylabel('coefficients')
        fig.savefig(output_dir + '/inference/rollout_epoch{:d}.png'.format(epoch), bbox_inches='tight')
        pick_channel = 1

    tqdm.write('{:e}'.format(temp_error))