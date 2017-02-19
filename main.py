import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import images
import convolves
import filters





def main():

    #read image
    im = plb.imread("P1012538.JPG")

    #Computing PSF
    h = convolves.gaussian(5, 11, 11)

    #Convolve images
    con=convolves.convolution_and_noise_rgb(im,h, np.zeros((2,2)))

    #Deblurring
    filt = filters.inverse_filter_rgb(con, h)

    #Show images
    plt.figure()
    plt.subplot(1, 4, 1)
    plt.imshow(im)
    plt.subplot(1, 4, 2)
    plt.imshow(np.uint8(con))
    plt.subplot(1, 4, 3)
    plt.imshow(np.uint8(filt))
    plt.subplot(1, 4, 4)
    plt.imshow(h, cmap='gray')
    plt.show()

    #Compare deblurred images and original
    filt = filt[0:im.shape[0], 0:im.shape[1]]
    print 'diff = ', images.compare_images_rgb(filt, im)

    #Saving deblurred images
    plt.imsave("inverse_filter/P1012538_inverse.jpg", np.uint8(filt))

if __name__ == "__main__":
    main()