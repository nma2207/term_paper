import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import images
import convolves
import filters





def main():

    im = plb.imread("P1012538.JPG")
    h = convolves.gaussian(1, 3, 3)
    con=convolves.convolution_and_noise_rgb(im,h, np.zeros((2,2)))
    filt = filters.inverse_filter_rgb(con, h)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(im)
    plt.subplot(1, 3, 2)
    plt.imshow(con)
    plt.subplot(1, 3, 3)
    plt.imshow(filt)
    plt.show()
    plt.imsave("inverse_filter/res2.jpg", filt)

if __name__ == "__main__":
    main()