% Faster R-CNN testing file for Object Recognition (Land marks detection for nevigation).
% All rights are reserved by Delowar Hossain
% E-mail: delowar_cse_ru@yahoo.com
Date: 22-06-2018
%% Clear the table
clear;
close all;
obj=videoinput('winvideo',2,'YUY2_640x480');  % create video input device
set(obj,'TriggerRepeat',inf) % set graphics object properties
set(obj,'ReturnedColorSpace','rgb')
start(obj)
%% Load R-CNN
% [net_loc, net_fol] = uigetfile({'*.mat','Binary MATLAB Files';...
%           '*.*','All Files' },'Browse The Network File');
% net_fil = fullfile(net_fol,net_loc);
% net = load(net_fil,'objFRC');
% net = net.objFRCNN;
load lmFRC_SNet_OL1
net = lmFRCNN;
%% Get image file
frame=getdata(obj,1);
test_im = frame;
imshow(test_im)
%figure
stop(obj)
%test_im = imread(file_nam);
%% Detection
tic
[bboxes, score, label] = detect(net, test_im)
toc
%% Output image
ixx = 1;
for i=1:size(score)
    if score(i)>=0.5
        bbox(ixx,:) = bboxes(i,:);
        label_str{ixx} = char(string(label(i)));
        ixx = ixx+1;
    end
end
if exist('bbox','var')
    outputImage = insertObjectAnnotation(test_im, 'rectangle', bbox, label_str,'FontSize', 14,'LineWidth',3);
    figure
    imshow(outputImage)
    title('Object Detected')
else
    figure
    imshow(test_im)
    title('Nothing')
end