import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_regular_transformer(model, device, train_loader, optimizer, criterion, epoch = 0, scheduler = None , x_normalizer = None, output_dir = None, save_interval = 100):
    model.train()
    train_ins_error = []
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['input']
        labels = batch['output']
        if x_normalizer is not None:
            inputs = x_normalizer.normalize(inputs)
            labels = x_normalizer.normalize(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        loss = criterion(output[:,-1,:], labels[:,-1,:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch % save_interval == 0) and batch_idx ==1:
            pick_channel = 0
            # plot out the enery distribution
            fig, ax = plt.subplots(figsize=(10,8))
            lw=1
            ax.plot(inputs[0,:, pick_channel].detach().cpu().numpy(),color = 'C0',linestyle='solid',linewidth=lw,alpha=1,label='input')
            ax.plot(labels[0,:, pick_channel].detach().cpu().numpy(),color = 'C1',linestyle='solid',linewidth=lw,alpha=1,label='label')
            ax.plot(output[0,:, pick_channel].detach().cpu().numpy(),color = 'C2',linestyle='solid',linewidth=lw,alpha=1,label='output')
            leg = ax.legend(loc='lower right', frameon = True, bbox_to_anchor=(1,1), ncol = 3)
            leg.get_frame().set_edgecolor('black')
            ax.set_xlabel('time')
            ax.set_ylabel('coefficients')
            ax.set_title(str(loss.item()))
            fig.savefig(output_dir+'/figs/temp_data_epoch{:d}.png'.format(epoch),bbox_inches='tight' )

        if scheduler != None:
            scheduler.step()
        train_ins_error.append(loss)

    return torch.mean(torch.stack(train_ins_error))


def train_regular_transformer_autoregress(model, device, train_loader, optimizer, criterion, forward, epoch=0,
                                          scheduler=None, x_normalizer=None, output_dir=None, save_interval=100):
    model.train()
    train_ins_error = []

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['input']
        labels = batch['output']

        # if x_normalizer is not None:
        #     inputs = x_normalizer.normalize(inputs)
        #     labels = x_normalizer.normalize(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Initialize hidden states if needed for transformer (depends on model)
        # If your model uses hidden states, you would need to reset them here, otherwise skip.

        # Use a temporary variable for autoregressive predictions to avoid corrupting the original `inputs`
        temp_inputs = inputs.clone()

        # Autoregressive prediction: predict the next `forward` steps
        for i in range(forward):
            output = model(temp_inputs)  # Assuming model outputs the prediction for the current time step

            # Autoregressive step: concatenate the last predicted output to the input for next prediction
            temp_inputs = torch.cat([temp_inputs[:, :, :], output[:, -1:, :]], dim=1)

        # Calculate loss: we use the last `forward` steps to calculate the loss against the labels
        loss = criterion(temp_inputs[:, -forward:, :], labels[:, -forward:, :])

        # Zero gradients, backward pass, optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Plot the input, label, and output only once per epoch or batch
        if (epoch % save_interval == 0) and batch_idx == 1:

            pick_channel = 0  # Pick a channel to visualize (adjust as needed)
            fig, ax = plt.subplots(figsize=(10, 8))
            lw = 1

            # Plot the original input sequence for the first batch, first channel
            ax.plot(inputs[0, :, pick_channel].detach().cpu().numpy(), color='C0', linestyle='solid', linewidth=lw,
                    alpha=1, label='input')

            # Plot the true labels sequence (shifted by forward_steps)
            ax.plot(labels[0, :, pick_channel].detach().cpu().numpy(), color='C1', linestyle='solid', linewidth=lw,
                    alpha=1, label='label')

            # Plot the last predicted output (after the autoregressive loop)
            ax.plot(temp_inputs[0, :, pick_channel].detach().cpu().numpy(), color='C2', linestyle='solid', linewidth=lw,
                    alpha=1, label='predicted_output')

            # Add legend
            leg = ax.legend(loc='lower right', frameon=True, bbox_to_anchor=(1, 1), ncol=3)
            leg.get_frame().set_edgecolor('black')

            # Set plot labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Coefficients')
            ax.set_title(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Save the plot as a PNG file
            fig.savefig(f"{output_dir}/figs/temp_data_epoch{epoch}.png", bbox_inches='tight')

        # Scheduler step if provided
        if scheduler is not None:
            scheduler.step()

        # Collect training error for monitoring
        train_ins_error.append(loss)

    # Return the average loss over the epoch
    return torch.mean(torch.stack(train_ins_error))


def test_rollout_autoregress(model, device, data, seq_len = 1,  x_normalizer = None):
    model.eval()
    predicted_data = data[:seq_len]

    for i in tqdm(range(len(data)-seq_len)):
        output = model(data[i:i+seq_len].unsqueeze(0).to(device)) # <B,L,C>
        predicted_data = torch.cat((predicted_data,output[0,-1].unsqueeze(-1).detach().cpu()), dim = 0)

    predicted_data_denorm = x_normalizer.denormalize(predicted_data)
    gt_data_denorm = x_normalizer.denormalize(data)

    return predicted_data_denorm, gt_data_denorm


def test_rollout(model, device, data, seq_len = 1,  x_normalizer = None):
    model.eval()
    predicted_data = data[:seq_len] # x_normalizer.normalize(data[:seq_len])
    gt_data = data[:seq_len]
    # print(predicted_data.shape, gt_data.shape) #<100,1> <100,1>
    for i in tqdm(range(len(data)-seq_len)):
        output = model(predicted_data[-seq_len:].unsqueeze(0).to(device)) # <B,L,C>
        predicted_data = torch.cat((predicted_data,output[0,-1].unsqueeze(-1).detach().cpu()), dim = 0)
        # print('in tests',gt_data.shape, data[seq_len+i].unsqueeze(-1).detach().cpu().shape)
        gt_data = torch.cat((gt_data, data[seq_len+i].unsqueeze(-1).detach().cpu()), dim = 0)
    predicted_data_denorm =  x_normalizer.denormalize(predicted_data)
    gt_data_denorm = x_normalizer.denormalize(gt_data)
    return predicted_data_denorm, gt_data_denorm
