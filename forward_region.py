#!/usr/bin/env python

# encoding: utf-8
import numpy as np

def forward_region(layer_in,size_in,maps_in):
    width=size_in[1]
    height=size_in[0]
    coords=4
    unknown=1
    classes=80
    layer_out=layer_in
    size_out=size_in
    maps_out=maps_in
    for n in range(5):
        id_start=n*width*height*(coords+unknown+classes)
        id_end=n*width*height*(coords+unknown+classes)+2*width*height
        layer_out[id_start:id_end] = 1. / (1 + np.exp(-layer_in[id_start:id_end]))
        id_start = n * width * height * (coords + unknown + classes) +coords * width * height
        id_end = n * width * height * (coords + unknown + classes) +coords * width * height + width * height
        layer_out[id_start:id_end] = 1. / (1 + np.exp(-layer_in[id_start:id_end]))
        id_start = n * width * height * (coords + unknown + classes) +(coords + unknown) * width * height
        id_end = n * width * height * (coords + unknown + classes) +(coords + unknown) * width * height + classes * width * height
        mat=layer_in[id_start:id_end].reshape(width,height,classes,order="F")
        largest=mat.max(2)
        for i in range(classes):
            mat[:,:,i]=np.exp(mat[:,:,i]-largest)
        summation=np.sum(mat,axis=2)
        for i in range(classes):
            mat[:,:,i]=mat[:,:,i]/summation
        layer_out[id_start:id_end]=mat[:].reshape(3920,1,order="F")
    return layer_out,size_out,maps_out

from keras.utils.vis_utils import plot_model