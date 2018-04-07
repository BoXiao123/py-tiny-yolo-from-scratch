#!/usr/bin/env python

# encoding: utf-8
import numpy as np

def img2col(data_im,channels,height,width,ksize,stride,pad):
    height_col=(height+2*pad-ksize)/stride+1
    width_col=(width+2*pad-ksize)/stride+1
    channels_col=channels*ksize*ksize
    data_col=np.zeros(shape=(height_col*width_col*channels_col,1))
    for c in range(channels_col):
        w_offset=np.mod(c,ksize)
        h_offset=np.mod(np.floor(c/ksize),ksize)
        c_im=np.floor(np.floor(c/ksize)/ksize)
        for h in range(height_col):
            for w in range(width_col):
                im_row=h_offset+h*stride
                im_col=w_offset+w*stride
                col_index=(c*height_col+h)*width_col+w
                data_col[col_index]=im2col_get_pixel(data_im,height,width,im_row,
                                                     im_col,c_im,pad)
    return data_col

def im2col_get_pixel(im,height,width,row,col,channel,pad):
    row=row-pad
    col=col-pad
    if row<0 or col<0 or row>=height or col>=width:
        pixel=0
    else:
        pixel=im[int(col+width*(row+height*channel))]
    return pixel
