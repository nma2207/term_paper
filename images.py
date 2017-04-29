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
    a_r=a[:,:,0]
    a_g=a[:,:,1]
    a_b=a[:,:,2]

    b_r=b[:,:,0]
    b_g=b[:,:,1]
    b_b=b[:,:,2]

    return np.array([compare_images(a_r,b_r),
                     compare_images(a_g, b_g),
                     compare_images(a_b, b_b)])

def correct_image(im):
    i=0
    f_im=np.copy(im)
    for i in range(f_im.shape[0]):
        for j in range(f_im.shape[1]):
            if(f_im[i,j]<0):
                f_im[i,j]=0
            if(f_im[i,j]>255):
                f_im[i,j]=255
    return f_im

def correct_image_rgb(f):
    f_r=f[:,:,0]
    f_g=f[:,:,1]
    f_b=f[:,:,2]
    res_r=correct_image(f_r)
    res_g = correct_image(f_g)
    res_b = correct_image(f_b)
    result=np.zeros((res_r.shape[0], res_r.shape[1],3))
    result[: ,:, 0] = res_r
    result[:, :, 1] = res_g
    result[:, :, 2] = res_b
    return result

def check_image(f):
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if f[i,j]<0 or f[i,j]>255:
                print 'All is bad:',i,j,f[i,j]
