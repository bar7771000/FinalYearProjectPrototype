# FinalYearProjectPrototype

As a part of my Final year project I am developing a desktop application which is able to upload an image from a local device, add a different types of noise to it and perform a prediction using pre-trained YOLOv4 model.
Noise and distortion very often appear in digital images, during the process of transmission and processing and without adequate knowledge of a noise model, it can be fairly difficult thing to remove. Having that in mind, the aim of this project is to implement and develop a desktop application, that helps to document a change that occurs in person recognition, by pre-trained neural network model, between original and distorted picture. The deformity will be performed by using different noise types, which include: Gaussian, Poison, Salt and Pepper etc. The distinction of predictions will be then compared using a diagram, including amount and type of noise added, for the purpose of determine what kind of noise could influence prediction the most and what is the threshold of the distortion when model will not be capable of recognising a person no more. 

----
To use the application, it is required to add:
-model configuration (https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg) and model weights (https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) files to main application file.
Model is trained on COCO dataset which allows it to detect up to 80 different classes.

----
YOLOv4 Paper - https://arxiv.org/pdf/2004.10934.pdf
COCO Dataset Paper - https://arxiv.org/pdf/1405.0312.pdf


