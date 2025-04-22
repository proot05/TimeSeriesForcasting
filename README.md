# TIme Series Forcasting for On-the-Fly 3D Printer Control

## Description
The goal of this project is to improve the LSTM algorithm a six-axis 3D printer uses to predict the state of the surface 
to be printed on. To begin this process, a high resolution, low frequency depth camera is used to capture high quality 
point clouds of the surface at various stages in its relatively periodic motion. The point clouds are processed into high 
quality meshes, soft correspondence is preformed on them to link points that represent physical locations on the surface 
between the different mesh states, and the scans are interpolated between to increase the spatial resolution of the possible 
mesh states. A pattern to be printed is then designed on the mesh state with the least deformation and this pattern is projected 
onto all the mesh states. Immediately prior printing, a low resolution, high frequency depth camera is used to monitor the current state
of the surface. The camera captures a point cloud of the mesh state, preforms some postprocessing, and finds the id of the high quality
mesh that best represents the current surface state (has the lowest chamfer loss). The printer observes the shape change of the surface 
for a period of time, creating a history of mesh states (ids) the surface goes through during that period. That data is used to train a
model with one LSTM layer and two linear layers (the first of which has an activation function on its output) to predict a mesh state a 
time into the future given an input window of the history of mesh states up to that point.

## Datasets
The main datasets used in this project, found in the datasets folder, are time histories of the surface id collected during printing trials.
There are two different collections of data, one corresponding to the surface state changing with a dominant frequency of 0.67 Hz and one with a
dominant frequency of 2 Hz. Within the data folders and files there is additional collected data and plots representing different evaluation criteria
of the trial, but we will only be utilizing the surface state time history within the lists.pkl files. Within the surface state time histories, there
are three distinct sections of collected data corresponding to different depth camera sampling rates. The first section of data corresponds to the highest
camera sampling rate (~ 40 Hz) and is the data the model will be trained on for a printing trial. The next section is data collected while predictions are being made, 
and therefore the camera sampling rate (~ 18 Hz) has decreased due to the computational load. The final section is data collected while predictions are being made and
commands are being sent to the printer, and therefore the camera sampling rate  (~ 10 Hz) has decreased again due to the additional computational load.
In application, the algorithm will be trained on the data collected at the high sampling rate (far above Nyquist criterion) and needs to make accurate predictions
on data at a lower sampling rate (still above Nyquist criterion), which presents a challenge.

In addition to data collected at different sampling rates and from different trials (both with different and the same surface frequency), the algorithm will be
tested on data with completely different periodic form than the training data. This data will come from the pydicom example time series datasets, which includes
electrocardiogram (ECG), respiratory, and audio waveforms collected at a constant sampling frequency.

## Install
```
conda create --name TSF python=3.9
conda activate TSF
pip install --index-url https://download.pytorch.org/whl/cu116  --no-deps torch==1.13.0+cu116
pip install -r requirements.txt
```

## Performance Metrics


## Model Training Workflow 
The data for training is first resampled to a constant time step time base and is zero mean and unit variance normalized. The data is then divided into batches,
where the input is a sequence length samples of the data and its label is the same collection of data plus a number of additional samples into the future. For training,
the model receives the sequence length input of original data and is autoregressively (adding an output one time step into the future onto the end of the input data)
used to make the same number of predictions into the future as the labels were established with. The loss is then calculated on all the future autoregressive outputs produced
for the input sequence length of original data. The loss function is a combination of mean squared error (MSE), a convolutional first derivative (to capture more 
information regarding the performance around low frequency features), and a dilated convolutional first derivative (to capture more information regarding the performance 
around high frequency features). Every certain number of epochs, the performance metrics of the model are evaluated on the training data by feeding in a rolling sequence length
input window of the data and predicting one time step into the future to effectively recreate the ground truth data. In implementation, the time into the future the model needs 
to predict the surface state at is not constant and depends on different hardware and software limitations. This is accounted for during printing by utilizing the cross correlation 
of the printer position history and printing surface state history to adjust the time into the future to predict. Therefore, the model is trained to autoregressively make predictions 
0.4 s into the future, which is larger than the prediction window typically seen during trials (0.15 - 0.35 s).

## Model Validation Workflow 
For validation, data in the same form as the training data, but from a different sample, is utilized. The data is implemented as it would be in on-the-fly printer control, giving a 
rolling sequence length input window of data to the model. The window of data is resampled to have the same time step as the resampled training data and is normalized using the 
same zero mean and unit variance as the training data. The model is used to autoregressively make predictions a constant time into the future, 0.25 s is what it is currently set to 
and was used for the plots in the two update sections (various prediction times are evaluated in the discussion section). If the time into the future to predict is not a multiple of the 
resampled time step, an extra autoregressive output is produced and the predicted point is linearly interpolated between the last two outputs.

## Run Instructions
To run the trained neural network on a validation sample, open and run the TestLSTM1.py file. It will run the predictions in the manner detailed in the last section and plot the ground truth
data, predicted data, and error as a function of time, with the performance metrics listed in the title. The plots, one of the entirety of the data and one of the last six seconds of data, will 
be saved in the \TimeSeriesForcasting\testfuncs\test_output\LSTM1 folder.

## First Update
The initial backbone of the model developed in this project was started by a former student in my lab and was completed and improved upon by myself. This first update represents the 
combination of our efforts and the state of the model midway through the semester. This model does not use autoregressive outputs in the training, instead only computing the loss on one
resampled time step into the future. The loss function was only MSE and it used a relu activation function after the first linear layer. Additionally, the model was found to improperly 
handle normalization and have a poor method of interpolating predicted data at predictions times into the future that were not multiples of the resampled time step.

## Current Model Performance Update
This model uses the exact training and validation workflows detailed above. Additionally, it uses a tanh activation function after the first linear layer and properly handles normalization and 
output interpolation.

## Discussion