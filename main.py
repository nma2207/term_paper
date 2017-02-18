import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import images
import convolves
import filters





def main():

    im = plb.imread("P1012538.JPG")
#    scipy.ndimage.gaussian_filter()

    bw=images.make_gray(im)
    print bw
    h = filters.gauss_filter(300 ,50, 50)
    #h=np.ones((7,7))*(1./7**2)
    print 'h=',h
    con = convolves.convolution(bw, h, np.array([[0, 0], [0, 0]]))
    print 'con'
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(bw, cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(con, cmap='gray')
    # plt.show()
    #print con
    print 'go'
    filt=filters.inverse_filter(con,h)
#    filt=wiener_filter(bw,h, np.array([[1,1],[1,1]]))
    print 'tickhonov'
    #filt=tickhonov_regularization(con,h)
    #bwi=np.zeros((con.shape[0],con.shape[1],3))
    #plt.imsave("convolution/res6.jpg", bwi)
    print 'filt'

    plt.figure()

    filt=filt[0:bw.shape[0],0:bw.shape[1]]

    origin = plb.imread("D:\\blurred_image\\CERTH_ImageBlurDataset\\EvaluationSet\\DigitalBlurSet\\GaussianH50x50S300_29.jpg")

    origin_bw=images.make_gray(origin)
    #filt=inverse_filter(origin_bw, h)
    #con = con[1:513, 1:513]
    plt.subplot(2, 2, 1)
    #plt.imshow(con, cmap='gray')
    plt.subplot(2, 2, 2)
    #plt.imshow(filt, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(origin_bw, cmap='gray')
    plt.subplot(2, 2, 4)
    filt = filters.inverse_filter(origin_bw,h)
    plt.imshow(filt, cmap='gray')
    plt.show()
    #plt.imsave("inverse_filter/res1.jpg", bwi)

   # print 'dif_old=',comp_image(origin_bw, bw)
    #print 'dif_new=', comp_image(con ,origin_bw)
if __name__ == "__main__":
    main()