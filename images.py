import numpy as np

#You can norm convert, if your image gave RGB-format
def make_gray(pix):
    h=pix.shape[0]
    w=pix.shape[1]
    res=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            res[i,j]=np.sum(pix[i,j])
    res=res//3
    return res

#You can compare if images have identical size
def compare_images(a,b):
    return np.sum((a-b)**2)

def compare_images_rgb(a,b):
    print 'TODO!!!'