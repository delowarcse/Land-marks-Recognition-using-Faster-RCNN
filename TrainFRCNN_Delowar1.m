% Faster R-CNN training file for Object Recognition (Land marks detection for nevigation).
% All rights are reserved by Delowar Hossain
% E-mail: delowar_cse_ru@yahoo.com
Date: 18-06-2018
%% System clear
close all
clear all
clc
%% Load data
% Call the UI for file browsing
[data_loca,data_path] = uigetfile({'*.mat','Binary MATLAB Files';...
          '*.*','All Files' },'Browse Data File');
data_file = fullfile(data_path,data_loca);
% Load the data
tmp = load(data_file,'LBT2');
data = tmp.LBT2;
%% Network training
% Check if the network is existing
if exist('lmFRC_SNet_02062018.mat','file')
    load('lmFRC_SNet_02062018.mat','lmFRCNN')
else
    % If not, create the new network.
    % Set input layer, size should be similar to smallest object.
    inputLayer = imageInputLayer([64 64 3]);
    % Middle layers.
    % Set convolutional layer parameters.
    filterSize = [3 3];    
    numFilters = 32;
    filterSize2 = [5 5];    
    numFilters2 = 64;
    % Set middle layers.
    middleLayers = [                
        convolution2dLayer(filterSize, numFilters, 'Padding', 1)
        reluLayer()
        convolution2dLayer(filterSize2, numFilters2, 'Padding', 2)
        reluLayer()
        maxPooling2dLayer(3, 'Stride',2)
        ];
    % Final layers
    finalLayers = [    
        % Add a fully connected layer with 64 output neurons. The output size
        % of this layer will be an array with a length of 64.
        fullyConnectedLayer(128)
        % Add a ReLU non-linearity.
        reluLayer()
        % Add the last fully connected layer. At this point, the network must
        % produce outputs that can be used to measure whether the input image
        % belongs to one of the object classes or background. This measurement
        % is made using the subsequent loss layers.
        fullyConnectedLayer(width(data))
        % Add the softmax loss layer and classification layer. 
        softmaxLayer()
        classificationLayer()
        ];
    layers = [
        inputLayer
        middleLayers
        finalLayers
        ]
    % Configure training options
    % Options for step 1.
    optionsStage1 = trainingOptions('sgdm', ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-4, ...
        'CheckpointPath', tempdir);
    % Options for step 2.
    optionsStage2 = trainingOptions('sgdm', ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-5, ...
        'CheckpointPath', tempdir);
    % Options for step 3.
    optionsStage3 = trainingOptions('sgdm', ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 256, ...
        'InitialLearnRate', 1e-4, ...
        'CheckpointPath', tempdir);
    % Options for step 4.
    optionsStage4 = trainingOptions('sgdm', ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 1e-5, ...
        'CheckpointPath', tempdir);
    options = [
        optionsStage1
        optionsStage2
        optionsStage3
        optionsStage4
        ];
    lmFRCNN = trainFasterRCNNObjectDetector(data, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'BoxPyramidScale', 1.2);
    net_name = 'lmFRC_SNet_02062018.mat';
    save(net_name, 'lmFRCNN');
end