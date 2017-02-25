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
    noise=convolves.add_normal_noise_rgb(im, 0, 5)

    #Deblurring
    #filt = filters.inverse_filter_rgb(con, h)
    filt = filters.inverse_filter_rgb(con, h)
    #filt = filters.tickhonov_regularization_rgb(con, h)

    #Show images
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(im)
    plt.title('original')
    plt.subplot(1, 4, 2)
    plt.imshow(np.uint8(con))
    plt.title('gaussian \n sigma=30, 11x11')
    plt.subplot(1, 4, 3)
    plt.imshow(np.uint8(filt))
    plt.title('inverse filter')
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
    h1=convolves.motion_blur(4, 0)
    h2=convolves.motion_blur(4,45)
    h3=convolves.motion_blur(4,90)
    print 'h1='
    print h1
    print 'h2='
    print h2
    print 'h3='
    print h3
if __name__ == "__main__":
   main()