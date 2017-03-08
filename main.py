import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import images
import convolves
import filters
import math
import threading
import time
from multiprocessing import Pool
from multiprocessing import Process




def main():

    #read image
    im = plb.imread("original/P1012538.JPG")

    #Computing PSF
    h = convolves.gaussian(30, 11, 11)
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
    h=convolves.motion_blur(10,90)
    con=convolves.convolution_rgb(im, h)
    print con.shape
    print im.shape
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
    filt=filters.lucy_richardson_devonvolution_rgb(con, h,500)
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(np.uint8(im))
    plt.title('original')
    plt.subplot(1,4,2)
    plt.imshow(np.uint8(con))
    plt.title('Motion blur\nlen=20 angle=30\nnoise~N(0,2)')
    plt.subplot(1,4,3)
    plt.imshow(np.uint8(images.correct_image_rgb(filt)))
    plt.title('Lucy-Richardson\ndeconvolution\neps=500')
    plt.subplot(1,4,4)
    plt.imshow(h,cmap='gray')
    plt.title('PSF')
    plt.show()

def sum(x):
    for i in range(100):
        a1=np.random.rand(x,x)
        a2=np.random.rand(x,x)
        a1*a2

    #print a1*a2

def test2():
    values=[2000,2500,3000]
    pool=Pool()
    pool.map(sum, values)
    pool.close()
    pool.join()

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
    sum(2000)
    sum(2500)
    sum(3000)


if __name__ == "__main__":
    start1=time.time()
    for i in range(10):
        print i," 1"
        test2()
    end1=time.time()
    start2=time.time()
    for i in range(10):
        print i," 2"
        test3()
    end2=time.time()
    print (end1-start1)/10.
    print (end2-start2)/10.