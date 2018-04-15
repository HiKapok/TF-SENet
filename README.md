# SE\_ResNet and SE\_ResNeXt model for Tensorflow with pre-trained weights on ImageNet

This repository contains code of the **un-official** re-implement of SE\_ResNe?t50 and SE\_ResNe?t101 model. Here is the authors' [implementation](https://github.com/hujie-frank/SENet) in Caffe.

SENet is one state-of-the-art convolutional neural network architecture, where dynamic channelwise
feature recalibration have been introduced to improve the representational capacity of CNN. More details can be found in the original paper: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf). 
##  ##
In order to accelerate further research based on SE\_ResNet and SE\_ResNeXt, I would like to share the pre-trained weights on ImageNet for them, you can download from [Google Drive](https://drive.google.com/open?id=1k5MtfqbNRA8ziE3f18vu00Q1FQCzk4__). The pre-trained weights are converted from [official weights in Caffe](https://github.com/hujie-frank/SENet) using [MMdnn](https://github.com/Microsoft/MMdnn) with other post-processing. And the outputs of all the network using the converted weights has almost the same outputs as original Caffe network (errors<1e-5). All rights related to the pre-trained weights belongs to the original author of [SENet](https://github.com/hujie-frank/SENet).

**This code and the pre-trained weights only can be used for research purposes.**

The canonical input image size for this SE\_ResNe?t is 224x224, each pixel value should in range [-128,128](BGR order), and the input preprocessing routine is quite simple, only normalization through mean channel subtraction was used. According to the official open-source  version in Caffe, SE-ResNe?t models got to 22.37%(SE-ResNet-50) and 20.97%(SE-ResNeXt-50) on ImageNet-1k for the single crop top-1 validation error.

The codes was tested under Tensorflow 1.6, Python 3.5, Ubuntu 16.04. 

BTW, other scaffold need to be build for training from scratch. You can refer to [resnet/imagenet_main](https://github.com/tensorflow/models/blob/22ded0410d5bed85a88329e852cd20882593652b/official/resnet/imagenet_main.py#L189) for adding weight decay to the loss manually.
##  ##
Apache License 2.0
