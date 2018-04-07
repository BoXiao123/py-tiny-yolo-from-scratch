#py-tiny-yolo-from-scratch

this repository is implementing tiny yolo by python from scratch. Yolo is a great and light weight deep learning platform. 
		https://pjreddie.com/darknet/yolo/

If you want to understand the full details of convolutional neural networks, you need to learn how to implement it from scatch. This code is based on the original codes of YOLO. And it is a python implementation. To sum up, it is a python vision of forward propogation of tiny yolo.

##Requirments
---
Numpy Opencv3

##Usage
---
python main.py

##Main parts
---
###main.py 
This is the pipeline of the tiny yolo. I have embeded the .cfg into the main.py.You can easily see the tiny yolo structure in this script.

###load_weights.py
I just follow the C code of YOLO about how to load the weights. You need to pay attention that the first 16 byte should be skipped. 16=4*sizeof(float)

###forward_convolution.py
Both of weights and feature maps are three dimensional. We need to pay attention in real programming ,there is no 3 dimensional structures for data. That is why we need to reshape the weights and input feature maps for each layer. The 2D convolution is like :

![](https://github.com/BoXiao123/py-tiny-yolo-from-scratch/raw/master/2.jpg)
	

