
import numpy as np
import cv2
import os
import json

def resize_pic(img, size):
    h,w,c = img.shape
    scale_factor = float(max(h,w)) / float(size)
    new_h = int(h/scale_factor)
    new_w = int(int(w/scale_factor))
    img_re = cv2.resize(img,(new_w,new_h))
    img_out = np.zeros(shape=[size,size,c],dtype=np.uint8)
    img_out[:new_h,:new_w,:] = img_re
    #cv2.imshow("img_out", img_out)
    return img_out

def trans_label_2_onehot(labels):  #lables is a list like [1,2,7,0,...], whose size is batch_size
    out = np.zeros(shape=[len(labels), 8],dtype=np.float32)
    for i in range(out.shape[0]):
        out[i][labels[i]] = 1.0
    return out




