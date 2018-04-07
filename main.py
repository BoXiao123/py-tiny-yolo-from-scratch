#!/usr/bin/env python

# encoding: utf-8
"""""
implemented by b_xi
this is a simple reimplemention of tiny-yolo
heavily inspired by the Matlab implementation

"""""

import numpy as np
import cv2
from init_layer import init_layer
from load_weights import load_weights
from forward_convolution import forward_convolution
from forward_max_pooling import forward_max_pooling
from forward_region import forward_region






if __name__=="__main__":
    #loading image
    print "loading image.."
    img=cv2.imread("1.jpg")
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    layer_in=init_layer(img)
    print "init image down"
    fid=open('tiny-yolo-coco.weights','rb')
    fid.seek(4*4)

    #layer1: conv 1*1*16
    print "layer 1 conv: ..."
    size_in=(img.shape[0],img.shape[1])
    maps_in=img.shape[2]
    maps_out=16
    kernel_size=3
    norm=1
    weights, biases, scales, rolling_mean, rolling_variance=\
        load_weights(fid,kernel_size,maps_in,maps_out,norm)
    layer_out,size_out=forward_convolution(layer_in,size_in,kernel_size,
            maps_in,maps_out,norm,weights,biases,scales,rolling_mean,rolling_variance)

    #np.savetxt("layer1_out.txt", layer_out, fmt='%.4f')

    #layer2: pool 1/2*1/2*16
    print "layer 2 (pool): ..."
    layer_in=layer_out
    size_in=size_out
    maps_in=maps_out
    kernel_size=2
    stride=2
    layer_out,size_out=forward_max_pooling(layer_in,size_in,maps_in,kernel_size,stride)
    #np.savetxt("layer2_out.txt", layer_out, fmt='%.4f')

    #Layer3: conv 1 / 2 * 1 / 2 * 32
    print "layer 3 (conv):..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    maps_out = 32
    kernel_size = 3
    norm = 1
    weights, biases, scales, rolling_mean, rolling_variance =load_weights(fid, kernel_size, maps_in, maps_out, norm)
    layer_out, size_out = forward_convolution(layer_in, size_in, kernel_size,
                                              maps_in, maps_out, norm, weights, biases, scales, rolling_mean, rolling_variance)

    # layer4: pool 1/4*1/4*32
    print "layer 4 (pool): ..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    kernel_size = 2
    stride = 2
    layer_out, size_out = forward_max_pooling(layer_in, size_in, maps_in, kernel_size, stride)


    # layer5: conv 1 / 4 * 1 / 4 * 64
    print "layer 5 (conv):..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    maps_out = 64
    kernel_size = 3
    norm = 1
    weights, biases, scales, rolling_mean, rolling_variance = load_weights(fid, kernel_size, maps_in, maps_out, norm)
    layer_out, size_out = forward_convolution(layer_in, size_in, kernel_size,
                                              maps_in, maps_out, norm, weights, biases, scales, rolling_mean,
                                              rolling_variance)

    # layer6: pool 1/8*1/8*64
    print "layer 6 (pool): ..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    kernel_size = 2
    stride = 2
    layer_out, size_out = forward_max_pooling(layer_in, size_in, maps_in, kernel_size, stride)


    # layer7: conv 1 / 8 * 1 / 8 * 128
    print "layer 7 (conv):..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    maps_out = 128
    kernel_size = 3
    norm = 1
    weights, biases, scales, rolling_mean, rolling_variance = load_weights(fid, kernel_size, maps_in, maps_out, norm)
    layer_out, size_out = forward_convolution(layer_in, size_in, kernel_size,
                                              maps_in, maps_out, norm, weights, biases, scales, rolling_mean,
                                              rolling_variance)

    # layer8: pool 1/16*1/16*128
    print "layer 8 (pool): ..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    kernel_size = 2
    stride = 2
    layer_out, size_out = forward_max_pooling(layer_in, size_in, maps_in, kernel_size, stride)

    # layer9: conv 1 / 16 * 1 / 16 * 256
    print "layer 9 (conv):..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    maps_out = 256
    kernel_size = 3
    norm = 1
    weights, biases, scales, rolling_mean, rolling_variance = load_weights(fid, kernel_size, maps_in, maps_out, norm)
    layer_out, size_out = forward_convolution(layer_in, size_in, kernel_size,
                                              maps_in, maps_out, norm, weights, biases, scales, rolling_mean,
                                              rolling_variance)

    # layer10: pool 1/32*1/32*256
    print "layer 10 (pool): ..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    kernel_size = 2
    stride = 2
    layer_out, size_out = forward_max_pooling(layer_in, size_in, maps_in, kernel_size, stride)


    # layer11: conv 1 / 32 * 1 / 32 * 512
    print "layer 11 (conv):..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    maps_out = 512
    kernel_size = 3
    norm = 1
    weights, biases, scales, rolling_mean, rolling_variance = load_weights(fid, kernel_size, maps_in, maps_out, norm)
    layer_out, size_out = forward_convolution(layer_in, size_in, kernel_size,
                                              maps_in, maps_out, norm, weights, biases, scales, rolling_mean,
                                              rolling_variance)

    #np.savetxt("layer11_out.txt", layer_out, fmt='%.4f')
    # layer12: pool 1/32*1/32*512
    print "layer 12 (pool): ..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    kernel_size = 2
    stride = 1
    layer_out, size_out = forward_max_pooling(layer_in, size_in, maps_in, kernel_size, stride)

    np.savetxt("layer12_out.txt", layer_out, fmt='%.4f')




    # layer13: conv 1 / 32 * 1 / 32 * 1024
    print "layer 13 (conv):..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    maps_out = 1024
    kernel_size = 3
    norm = 1
    weights, biases, scales, rolling_mean, rolling_variance = load_weights(fid, kernel_size, maps_in, maps_out, norm)
    layer_out, size_out = forward_convolution(layer_in, size_in, kernel_size,
                                             maps_in, maps_out, norm, weights, biases, scales, rolling_mean,
                                             rolling_variance)

    # layer14: conv 1 / 32 * 1 / 32 * 1024
    print "layer 14 (conv):..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    maps_out = 1024
    kernel_size = 3
    norm = 1
    weights, biases, scales, rolling_mean, rolling_variance = load_weights(fid, kernel_size, maps_in, maps_out, norm)
    layer_out, size_out = forward_convolution(layer_in, size_in, kernel_size,
                                             maps_in, maps_out, norm, weights, biases, scales, rolling_mean,
                                             rolling_variance)


    # layer15: conv 1 / 32 * 1 / 32 * 425
    print "layer 15 (conv):..."
    layer_in = layer_out
    size_in = size_out
    maps_in = maps_out
    maps_out = 425
    kernel_size = 1
    norm = 0
    weights, biases, scales, rolling_mean, rolling_variance = load_weights(fid, kernel_size, maps_in, maps_out, norm)
    layer_out, size_out = forward_convolution(layer_in, size_in, kernel_size,
                                             maps_in, maps_out, norm, weights, biases, scales, rolling_mean,
                                             rolling_variance)

    #layer 16: region
    print "layer 16 region: ..."
    layer_in=layer_out
    size_in=size_out
    maps_in=maps_out
    layer_out,size_out,maps_out=forward_region(layer_in,size_in,maps_in)
    #np.savetxt("layer16_out.txt", layer_out, fmt='%.4f')
    print "Done everything!"


