#!/usr/bin/env python

# encoding: utf-8
import numpy as np
from img2col import img2col

def forward_convolution(layer_in,size_in,kernel_size,
            maps_in,maps_out,norm,weights,biases,scales,rolling_mean,rolling_variance):
    width=size_in[1]
    height=size_in[0]
    size_out=[width,height]
    stride=1
    pad=(kernel_size-1)/2
    data_col=img2col(layer_in,maps_in,height,width,kernel_size,stride,pad)
    m=maps_out
    k=kernel_size*kernel_size*maps_in
    n=width*height
    A=weights.reshape(m,k)
    B=data_col.reshape(k,n)
    C=np.dot(A,B)
    layer_out=C.reshape(m*n,1)
    if norm:
        for i in range(maps_out):
            for j in range(width*height):
                index=i*width*height+j
                #normalization
                layer_out[index]=(layer_out[index]-rolling_mean[i])/(np.sqrt(rolling_variance[i])+0.000001)
                #scale and bias
                layer_out[index]=layer_out[index]*scales[i]
                layer_out[index]=layer_out[index]+biases[i]
                #activation
                if layer_out[index]<0:
                    layer_out[index]=layer_out[index]*0.1
    else:
        for i in range(maps_out):
            for j in range(width*height):
                index=i*width*height+j
                layer_out[index]=layer_out[index]+biases[i]
    print "Done! %d*%d*%d=>%d*%d*%d"%(width,height,maps_in,width,height,maps_out)
    return layer_out,size_out
