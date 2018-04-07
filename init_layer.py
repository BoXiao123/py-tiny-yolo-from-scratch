#!/usr/bin/env python

# encoding: utf-8
""""
reshape image
input is image
output is image which has been reshaped to N-by-1 Array 

"""""

import numpy as np

def init_layer(img):
    w,h,c= img.shape
    layer = np.zeros(shape=(2,1))
    for i in range(c):
        temp = img[:,:,i].reshape(w * h, 1)
        layer=np.concatenate((layer,temp),axis=0)
    layer=layer[2:,:]
    return layer/255
