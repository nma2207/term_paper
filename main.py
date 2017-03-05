import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import images
import convolves
import filters





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
    im = plb.imread("original/P1012538.JPG")
    h=convolves.averaging_filter(13,13)
    im=np.float64(im)
    con=convolves.convolution_rgb(im, h)
    con, noise=convolves.add_normal_noise_rgb(con, 0, 5)
    con=images.correct_image_rgb(con)
    filt=filters.lucy_richardson_devonvolution_rgb(con, h,100)
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(np.uint8(im))
    plt.title('original')
    plt.subplot(1,4,2)
    plt.imshow(np.uint8(con))
    plt.title('average bluring\n13x13\nnoise~N(0,5)')
    plt.subplot(1,4,3)
    plt.imshow(np.uint8(images.correct_image_rgb(filt)))
    plt.title('Lucy-Richardson\ndeconvolution\nn=100')
    plt.subplot(1,4,4)
    plt.imshow(h,cmap='gray')
    plt.title('PSF')
    plt.show()



if __name__ == "__main__":
   test1()