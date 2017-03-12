# coding=utf-8
import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import images
import convolves
import filters
import math
import threading
import time
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Process




def main():

    #read image
    im = plb.imread("original/P1012538.JPG")

    #Computing PSF
    h = convolves.motion_blur(10,30)
    #Convolve images
    con=convolves.convolution_rgb(im,h)
    noise=convolves.add_normal_noise_rgb(con, 0, 30)

    #Deblurring
    #filt = filters.inverse_filter_rgb(con, h)
    filt = filters.wiener_filter_rgb(con, h, noise, im)
    #filt = filters.tickhonov_regularization_rgb(con, h)

    #Show images
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(im)
    plt.title('original')
    plt.subplot(1, 4, 2)
    plt.imshow(np.uint8(images.correct_image_rgb(con)))
    plt.title('gaussian \n sigma=30, 11x11 \n noise~N(0, 30)')
    plt.subplot(1, 4, 3)
    plt.imshow(np.uint8(images.correct_image_rgb(filt)))
    plt.title('wiener filter')
    plt.subplot(1, 4, 4)
    plt.imshow(h, cmap='gray')
    plt.title('PSF')
    plt.show()

    #Compare deblurred images and original
    filt = filt[0:im.shape[0], 0:im.shape[1]]
    print 'diff = ', images.compare_images_rgb(filt, im)

    #Saving deblurred images
    plt.imsave("inverse_filter/P1012538_inverse.jpg", np.uint8(filt))

def test():
    im = plb.imread("original/P1012538.JPG")
    h=convolves.motion_blur(10,30)
    con=convolves.convolution_rgb(im, h)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.title('original')
    plt.subplot(1, 3, 2)
    plt.imshow(np.uint8(images.correct_image_rgb(con)))
    plt.title('Motion bluring\n len=50, ang=45')
    plt.subplot(1, 3, 3)
    plt.imshow(h, cmap='gray')
    plt.title('PSF')
    plt.show()

def test1():
    im = plb.imread("original/my_bike.jpg")
    h=convolves.motion_blur(20,30)
    im=np.float64(im)
    con=convolves.convolution_rgb(im, h)
    con, noise=convolves.add_normal_noise_rgb(con, 0, 2)
    con=images.correct_image_rgb(con)
    filt=filters.lucy_richardson_deconvolution_multythread(con, h,100)
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(np.uint8(im))
    plt.title('original')
    plt.subplot(1,4,2)
    plt.imshow(np.uint8(con))
    plt.title('Motion blur\nlen=20 angle=30\nnoise~N(0,2)')
    plt.subplot(1,4,3)
    plt.imshow(np.uint8(images.correct_image_rgb(filt)))
    plt.title('Lucy-Richardson\ndeconvolution\neps=50')
    plt.subplot(1,4,4)
    plt.imshow(h,cmap='gray')
    plt.title('PSF')
    plt.show()

def sum(x,y):
    for i in range(10):
        a1=np.random.rand(x,y)
        a2=np.random.rand(x,y)
        a1*a2
    return x

    #print a1*a2
def temp(t):
    return sum(*t)
def test2():
    #русские комментарии
    values=[(2000,1500),(2500,2000),(3000,3500)]
    pool=Pool(processes=mp.cpu_count())
    result=pool.map(temp, values)
    print result
    # pool.close()
    # pool.join()

    # p1 = Process(target=sum, args=(2000,))
    # p2 = Process(target=sum, args=(2500,))
    # p3 = Process(target=sum, args=(3000,))
    # p1.start()
    # p2.start()
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()


def test3():
    sum(2000,200)
    sum(2500,200)
    sum(3000,200)

def test_kosmichi():
    im = plb.imread("original/kosmichi.jpg")
    im = np.float64(im)
    h=convolves.motion_blur(20,100)
    filt=filters.lucy_richardson_devonvolution_rgb(im, h, 10000000)
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(np.uint8(im))
    plt.title('original')
    plt.subplot(1,3,2)
    plt.imshow(np.uint8(images.correct_image_rgb(filt)))
    plt.title('Lucy-Richardson\ndeconvolution\neps=10000')
    plt.subplot(1,3,3)
    plt.imshow(h,cmap='gray')
    plt.title('PSF')
    plt.show()




if __name__ == "__main__":
    test_kosmichi()
    # start1=time.time()
    # for i in range(1):
    #     print i," 1"
    #     test2()
    # end1=time.time()
    # start2=time.time()
    # for i in range(1):
    #     print i," 2"
    #     test3()
    # end2=time.time()
    # print 'multi-thread: ',(end1-start1)/1.
    # print 'posledovatelno: ',(end2-start2)/1.