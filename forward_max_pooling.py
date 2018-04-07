#!/usr/bin/env python

# encoding: utf-8
import numpy as np

def forward_max_pooling(layer_in,size_in,maps_in,p_size,p_stride):
    in_width=size_in[1]
    in_height=size_in[0]
    out_width=in_width/p_stride
    out_height=in_height/p_stride
    maps_out=maps_in
    layer_out=np.zeros(shape=(out_width*out_height*maps_out,1))
    size_out=[out_width,out_height]
    for m in range(maps_out):
        mat_in=layer_in[m*in_width*in_height:(m+1)*in_width*in_height].reshape(in_width,in_height)
        for h in range(out_height):
            for w in range(out_width):
                if h*p_stride+p_size>in_height or w*p_stride+p_size>in_width:
                    data=np.zeros(shape=(p_size,p_size))
                    for i in range(1,1+p_size):
                        for j in range(1,1+p_size):
                            if i+h*p_stride>in_height or j+w*p_stride>in_width:
                                data[i-1,j-1]=-9999
                            else:
                                data[i-1,j-1]=mat_in[i+h*p_stride-1,j+w*p_stride-1]
                else:
                    data=mat_in[h*p_stride:h*p_stride+p_size,w*p_stride:w*p_stride+p_size]
                layer_out[m*out_height*out_width+h*out_width+w]=np.max(np.max(data))
    print "Done! %d * %d * %d => %d * %d * %d "%(in_width, in_height, maps_in, out_width, out_height, maps_out)
    return layer_out,size_out

