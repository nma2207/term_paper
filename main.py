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
import matplotlib.mlab as mlab
import scipy.signal as sg
import scipy.misc as smisc



def main():

    #read image
    im = plb.imread("original/P1012538.JPG")

    #Computing PSF
    h = convolves.motion_blur(10,30)
    #Convolve images
    con=convolves.convolution_rgb(im,h)
    #noise=convolves.add_normal_noise_rgb(con, 0, 30)

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
    im = plb.imread("original/adelina.jpg")
    h=convolves.motion_blur(40,30)
   # h=convolves.gaussian(10,15,15)
    con=convolves.convolution_rgb(im, h)
    # plt.figure()
    # c=np.copy(con)
    # plt.imshow(np.uint8(images.correct_image_rgb(c)))
    # plt.show()
    con, noise=convolves.add_normal_noise_rgb(con, 0, 1)
    start=time.time()
    #filt=filters.tickhonov_regularization_rgb(con ,h,1e-2)
    #filt=filters.wiener_filter_rgb(con, h, K=1e-03)
    print 'go'
    #filt=filters.inverse_filter_rgb(con, h)
    #filt = filters.lucy_richardson_deconvolution_multythread(con ,h, 20000)
    filt = filters.tickhonov_regularization_rgb(con, h, 1e-3)
    #filt=filters.wiener_filter_rgb(con, h, K=1)
    print 'end'
    end=time.time()
    # f=filt[:im.shape[0],
    #         :im.shape[1],:3]
    f=filt[h.shape[0]//2:im.shape[0]+h.shape[0]//2,
            h.shape[1]//2:im.shape[1]+h.shape[1]//2,:3]
    #print im.shape, filt[:512,:512,:3].shape
    print images.compare_images_rgb(im, f)
    print end-start
    #np.set_printoptions(threshold=np.nan)
    print con
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(im)
    plt.title('original')
    plt.subplot(1, 4, 2)
    plt.imshow(np.uint8(images.correct_image_rgb(con)))
    #plt.title('Gaussian blur\nsigma=10, size=15x15\nnoise~N(0,1)')
    plt.title('Motion blur\nlen=40, ang=30\nnoise~N(0,1)')
    plt.subplot(1, 4, 3)
    plt.imshow(np.uint8(images.correct_image_rgb(filt)))
    plt.title('Weiner filter\nK = 1e-3')
    #plt.title('Tikhonov regulerization\ngamma = 1e-2')
    #plt.title('Inverse filter')
    #plt.title('Lucy-Richardson deconvolution\neps = 20000')
    plt.subplot(1, 4, 4)
    plt.imshow(h, cmap='gray')
    plt.title('PSF')
    plt.show()
    plt.figure()
    plt.imshow(np.uint8(images.correct_image_rgb(filt)))
    plt.show()

def test1():
    im = plb.imread("original/DSC02125.JPG")
    h=convolves.gaussian(1,512,512)
    im=np.float64(im)
    con=convolves.convolution_rgb(im, h)
    con, noise=convolves.add_normal_noise_rgb(con, 0, 2)
    con=images.correct_image_rgb(con)
    print 'go multy-thread'
    start1 = time.time()
    filt1=filters.lucy_richardson_deconvolution_multythread(con, h,20000)
    end1=time.time()
    print 'go posled'
    start2=time.time()
    filt2 = filters.lucy_richardson_devonvolution_rgb(con, h, 5000)
    end2=time.time()
    print 'multy-thread:', end1-start1
    print 'posledovatel:', end2-start2
    # plt.figure()
    # plt.subplot(1,4,1)
    # plt.imshow(np.uint8(im))
    # plt.title('original')
    # plt.subplot(1,4,2)
    # plt.imshow(np.uint8(con))
    # plt.title('Motion blur\nlen=20 angle=30\nnoise~N(0,2)')
    # plt.subplot(1,4,3)
    # plt.imshow(np.uint8(images.correct_image_rgb(filt)))
    # plt.title('Lucy-Richardson\ndeconvolution\neps=50')
    # plt.subplot(1,4,4)
    # plt.imshow(h,cmap='gray')
    # plt.title('PSF')
    # plt.show()

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



def test4():
    #начитавшись статей Южикова решил попробовать вычислять градиент
    #при помощи фильтра Собеля, т.к. вы говорили что эта штука дает карту ГРАДИЕНТОВ

    #Гистограмму считает оочень долго, поэтому не советую запускать

    im=plb.imread("original/gray_lena.jpg")
    im=images.make_gray(im)
    im=np.float64(im)
    h_x=convolves.sobel_filter_X()
    h_y=convolves.sobel_filter_Y()
    im_x=convolves.convolution(im,h_x)
    im_y=convolves.convolution(im, h_y)
    h=convolves.gaussian(10,51,51)
    con=convolves.convolution(im, h)
    con_x=convolves.convolution(con, h_x)
    con_y=convolves.convolution(con, h_y)
    im_g=np.sqrt(im_x**2+im_y**2)
    con_g=np.sqrt(con_x**2+con_y**2)

    im_x=im_x.flatten()
    im_y=im_y.flatten()
    print im_x.shape
    im_together=np.zeros((im_x.size+im_y.size))

    im_together[:im_x.size]=im_x
    im_together[im_x.size:im_x.size+im_y.size]=im_y

    con_x=con_x.flatten()
    con_y=con_y.flatten()
    con_together=np.zeros((con_x.size+con_y.size))
    con_together[:con_x.size]=con_x
    con_together[con_x.size:con_x.size+con_y.size]=con_y

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_g, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(con_g, cmap='gray')
    plt.show()
    plt.figure(1)
    print 'go'
    h1, bins1=np.histogram(im_together, bins=20)
    plt.plot(bins1[:-1], h1, 'r', label='original')
    h2, bins2=np.histogram(con_together, bins=20)
    plt.plot(bins2[:-1], h2, 'b', label='blurred')
    plt.legend()
    plt.show()
    # plt.subplot(1,2,1)
    # h1, bins1=np.histogram(im_x, bins=20)
    # h2, bins2 = np.histogram(con_x, bins=20)
    # plt.plot(bins1[:-1],h1, 'r', label='original')
    # plt.plot(bins2[:-1],h2, 'b', label='blurred')
    # plt.legend()
    # plt.title('Sobel - x')
    # plt.subplot(1,2,2)
    # h1, bins1=np.histogram(im_y, bins=20)
    # h2, bins2 = np.histogram(con_y, bins=20)
    # plt.plot(bins1[:-1],h1, 'r')
    # plt.plot(bins2[:-1],h2, 'b')
    # plt.plot(bins1[:-1],h1, 'r', label='original')
    # plt.plot(bins2[:-1],h2, 'b', label='blurred')
    # plt.legend()
    # plt.title('Sobel - y')
    # plt.show()


def test_l_r():
    im=plt.imread('original/lena.bmp')
    #gray=images.make_gray(im)
    #gray=np.float64(gray)
    im=np.float64(im)
    h=convolves.gaussian(5,35,35)
    print 'go con'
    con=convolves.convolution_rgb(im, h)
    con,noise=convolves.add_normal_noise_rgb(con, 0, 5)
    print 'go filt'
    filt=filters.lucy_richardson_deconvolution_multythread(con,h, 5000)
    con=np.uint8(images.correct_image_rgb(con))
    #con=convolves.add_normal_noise_rgb()
    filt=np.uint8(images.correct_image_rgb(filt))
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(np.uint8(im))
    plt.subplot(1,4,2)
    plt.imshow(con)
    plt.subplot(1,4,3)
    plt.imshow(filt)
    plt.subplot(1,4,4)
    plt.imshow(h, cmap='gray')
    plt.show()



def test_blind():
    im=plt.imread('original/lena.bmp')
    gray=images.make_gray(im)

    gray=np.float64(gray)
    #h=convolves.gaussian(13,15,15)
    h=convolves.motion_blur(20,30)
    #con=convolves.convolution2(gray, h)
    con=convolves.convolution2(gray, h)
    plt.imsave(fname='l_r_blind/real_h.bmp', arr=np.uint8(images.correct_image(h*255)), cmap='gray')
    filt,new_h=filters.lucy_richardson_blind_deconvolution0_1(images.make0to1(con),
                                                              50, 1,
                                                              images.make0to1(gray))
    plt.imsave(fname='l_r_blind/new_lena.bmp', arr=np.uint8( images.correct_image(filt)), cmap='gray')
    plt.figure()
    plt.subplot(1,5,1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original')
    plt.subplot(1,5,2)
    plt.imshow(np.uint8(images.correct_image(con)), cmap='gray')
    plt.title('Convoluton')
    plt.subplot(1,5,3)
    plt.imshow(np.uint8( images.correct_image(filt)), cmap='gray')
    plt.title('L-R blind\nn=20, m=10')
    plt.subplot(1,5,4)
    plt.imshow(h, cmap='gray')
    plt.title('PSF\ngaussian\nsigma=15, 13x13')
    plt.subplot(1,5,5)
    plt.imshow(new_h, cmap='gray')
    plt.title('new PSF')
    plt.show()
def test_zoom():
    im=plt.imread('original/lena.bmp')
    new_im=smisc.imresize(im, (50, 64))
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im)
    plt.subplot(1,2,2)
    plt.imshow(new_im)
    plt.show()

def test_pir():
    im=plt.imread('original/lena.bmp')
    gray=images.make_gray(im)

    gray=np.float64(gray)
    h=convolves.gaussian(13,15,15)
    #h=convolves.motion_blur(20,30)
    #print 'real_h =',h
    #con=convolves.convolution2(gray, h)
    con=convolves.convolution(gray, h)
    plt.imsave(fname='l_r_blind/real_h.bmp', arr=np.uint8(images.correct_image(h*255)), cmap='gray')
    filt,new_h=filters.lucy_richardson_blind_deconvolution_pir(con, 5, 1,1, 501, 'gaussian', original=gray)
    #filt=images.correct_image(filt*255)
    plt.imsave(fname='l_r_blind/new_lena.bmp', arr=np.uint8( images.correct_image(filt)), cmap='gray')
    plt.figure()
    plt.subplot(1,5,1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original')
    plt.subplot(1,5,2)
    plt.imshow(np.uint8(images.correct_image(con)), cmap='gray')
    plt.title('Convoluton')
    plt.subplot(1,5,3)
    plt.imshow(np.uint8( images.correct_image(filt)), cmap='gray')
    plt.title('L-R blind\nn=20, m=10')
    plt.subplot(1,5,4)
    plt.imshow(h, cmap='gray')
    plt.title('PSF\nmotion blur\nlen=20 ang=30')
    plt.subplot(1,5,5)
    plt.imshow(new_h, cmap='gray')
    plt.title('new PSF')
    #print images.compare_images(gray, con[:g])
    #print images.compare_images(gray, filt)
    plt.show()

def test_l_r_graph():
    im = plb.imread("original/my_bike.jpg")
    #h=convolves.motion_blur(20,30)
    gray=images.make_gray(im)
    h=convolves.gaussian(10,15,15)
    con=convolves.convolution(gray, h)
    #con, noise=convolves.add_normal_noise(con, 0, 1)
    start=time.time()
    print 'go'
    filt=filters.lucy_richardson_deconvolution(con, h ,eps=0, original= gray, N=82)
    print 'end'
    end=time.time()
    f=filt
    #print im.shape, filt[:512,:512,:3].shape
    #print images.compare_images_rgb(im, f)
    print end-start
    #np.set_printoptions(threshold=np.nan)
    print con
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('original')
    plt.subplot(1, 4, 2)
    plt.imshow(np.uint8(images.correct_image(con)), cmap='gray')
    plt.title('Gaussian blur\nsigma=10, size=15x15\nnoise~N(0,1)')
    #plt.title('Motion blur\nlen=20, ang=30\nnoise~N(0,1)')
    plt.subplot(1, 4, 3)
    plt.imshow(np.uint8(images.correct_image(filt)),cmap='gray')
    #plt.title('Tikhonov regulerization\ngamma=1e-02')
    plt.title('Lucy-Richardson deconvolution\neps=5000')
    plt.subplot(1, 4, 4)
    plt.imshow(h, cmap='gray')
    plt.title('PSF')
    plt.show()



if __name__ == "__main__":
    test()
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