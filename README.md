# Land marks recognitio using Faster R-CNN (regions with convolutional neural networks)
This Repository is for Land marks recognition purpose using Faster R-CNN on Robotics Applications

A Faster R-CNN object detection network is composed of a feature extraction network followed by two subnetworks. 
The feature extraction network is typically a pretrained CNN. 
The first subnetwork following the feature extraction network is a region proposal network (RPN) trained to generate object proposals - areas in the image where objects are likely to exist.
The second subnetwork is trained to predict the actual class of each object proposal.

TrainFRCNN_Delowar1.m trains a network using Faster R-CNN for Land marks recognition of input size [64 64 3] on our robotic applications.
TestFRCNN_Delowar2.m evaluates the trained land marks recognition on our robotic applications.
