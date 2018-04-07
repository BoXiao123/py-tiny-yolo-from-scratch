#!/usr/bin/env python
# encoding: utf-8
import array
import numpy as np

def load_weights(fid,kernel_size,maps_in,maps_out,norm):
    biases = array.array("f")
    biases.fromfile(fid, maps_out)
    biases = np.array(biases).reshape(maps_out, 1)
    if norm:
        scales=array.array("f")
        scales.fromfile(fid,maps_out)
        scales=np.array(scales).reshape(maps_out,1)
        print scales
        rolling_mean=array.array("f")
        rolling_mean.fromfile(fid,maps_out)
        rolling_mean=np.array(rolling_mean).reshape(maps_out,1)
        rolling_variance=array.array("f")
        rolling_variance.fromfile(fid,maps_out)
        rolling_variance=np.array(rolling_variance).reshape(maps_out,1)
    else:
        scales=0
        rolling_mean=0
        rolling_variance=0
    weights=array.array("f")
    weights.fromfile(fid,kernel_size*kernel_size*maps_in*maps_out)
    weights=np.array(weights).reshape(kernel_size*kernel_size*maps_in*maps_out,1)
    return weights,biases,scales,rolling_mean,rolling_variance

