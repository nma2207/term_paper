import numpy as np
import convolves as conv
import images
import multiprocessing as mp
from multiprocessing import Pool
from matplotlib import  pyplot as plt
import scipy.signal as sg
import scipy.misc as smisc
import math
import gc
def inverse_filter(g,h):
    width_g=g.shape[0]
    height_g=g.shape[1]
    width_h=h.shape[0]
    height_h=h.shape[1]
    g1=np.zeros((2*width_g,2*height_g))
    h1=np.zeros((2*width_g, 2*height_g))
    g1[0:width_g,0:height_g]=g
    h1[0:width_h, 0:height_h] = h
    G=np.fft.fft2(g1)
    H=np.fft.fft2(h1)
    F=G/H
    f=np.fft.ifft2(F)
    f=np.real(f)
    f=f[0:width_g,0:height_g]
    return f

def inverse_filter_rgb(g,h):
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    result=np.zeros(g.shape)
    result[:, :, 0] = inverse_filter(g_r, h)
    result[:, :, 1] = inverse_filter(g_g, h)
    result[:, :, 2] = inverse_filter(g_b, h)
    return result



def wiener_filter(g, h,K):
    width_g=g.shape[0]
    height_g=g.shape[1]
    width_h=h.shape[0]
    height_h=h.shape[1]
    g1=np.zeros((2*width_g,2*height_g))
    h1=np.zeros((2*width_g, 2*height_g))
    # n1=np.zeros((2*width_g, 2*height_g))
    # original1=np.zeros((2*width_g, 2*height_g))
    g1[0:width_g,0:height_g]=g
    h1[0:width_h, 0:height_h] = h
    # n1[0:n.shape[0],0:n.shape[1]]=n
    # original1[0:original.shape[0], 0:original.shape[1]] = original
    # N=np.fft.fft2(n1)
    # ORIGINAL=np.fft.fft2(original1)

    G=np.fft.fft2(g1)
    H=np.fft.fft2(h1)
    KK = np.ones(G.shape)*K
    print KK.shape
    print (np.abs(H) ** 2).shape
    #F = (np.abs(H) ** 2 / (H * ((np.abs(H) ** 2) ) )) * G
    F = (np.abs(H) ** 2 / (H * ((np.abs(H) ** 2)+K ))) * G
    print K
    f=np.fft.ifft2(F)
    f=np.real(f)
    f=f[0:width_g,0:height_g]
    return f

def wiener_filter_rgb(g,h,K=0):
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    # n_r=n[:,:,0]
    # n_g=n[:,:,1]
    # n_b=n[:,:,2]
    # original_r=original[:,:,0]
    # original_g=original[:,:,1]
    # original_b=original[:,:,2]
    result=np.zeros(g.shape)
    # result[:, :, 0] =  wiener_filter(g_r, h,n_r, original_r)
    # result[:, :, 1] =  wiener_filter(g_g, h,n_g, original_g)
    # result[:, :, 2] =  wiener_filter(g_b, h,n_b, K, original_b)
    result[:, :, 0] =  wiener_filter(g_r, h, K)
    result[:, :, 1] =  wiener_filter(g_g, h, K)
    result[:, :, 2] =  wiener_filter(g_b, h, K)
    return result

def tickhonov_regularization(g,h, gamma=0):
    p=np.array([
        [0,-1,0],
        [-1, 4, -1],
        [0,-1,0.0]
    ])
    width_g = g.shape[0]
    height_g = g.shape[1]
    width_h = h.shape[0]
    height_h = h.shape[1]
    g1 = np.zeros((2 * width_g, 2 * height_g))
    h1 = np.zeros((2 * width_g, 2 * height_g))
    g1[0:width_g, 0:height_g] = g
    h1[0:width_h, 0:height_h] = h
    G = np.fft.fft2(g1)
    H = np.fft.fft2(h1)
    p1=np.zeros((2 * width_g, 2 * height_g))
    p1[p1.shape[0]/2-1:p1.shape[0]/2+2,p1.shape[1]/2-1:p1.shape[1]/2+2]=p
    P=np.fft.fft2(p1)
    #gamma=0.
    F =(np.conjugate(H)/(np.abs(H)**2+gamma*np.abs(P)**2))*G
    f = np.fft.ifft2(F)
    f = np.real(f)
    f = f[0:width_g, 0:height_g]
    return f

def tickhonov_regularization_rgb(g,h, gamma=0):
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    result=np.zeros(g.shape)
    result[:, :, 0] = tickhonov_regularization(g_r, h, gamma)
    result[:, :, 1] = tickhonov_regularization(g_g, h, gamma )
    result[:, :, 2] = tickhonov_regularization(g_b, h, gamma)
    return result


#Filippov's article
def quick_blind_deconvolution(g):
    print 'TODO!!'


def lucy_richardson_deconvolution(g,h,eps =0, original=None, N=0):
    #n - iterations count
    f=np.copy(g)
    f_prev=np.zeros(f.shape, dtype=float)
    k=images.compare_images(f_prev, f)
    err=[]
    i=0
    while(True):

        f_prev=np.copy(f)

        k1=conv.convolution2(f,h)
        #k1=k1[h.shape[0]//2:f.shape[0]+h.shape[0]//2, h.shape[1]//2:f.shape[1]+h.shape[1]//2]
        k2=g/k1
        h1=np.flipud(np.fliplr(h))
        del k1

        k3=conv.convolution2(k2,h1)
        del k2
        del h1

        #k3 = k3[h.shape[0] // 2:f.shape[0] + h.shape[0] // 2, h.shape[1] // 2:f.shape[1] + h.shape[1] // 2]
        f=f*k3
        k_prev=k
        k=images.compare_images(f_prev, f)
        err.append(images.compare_images(f[h.shape[0]//2:original.shape[0]+h.shape[0]//2,
                                         h.shape[1]//2:original.shape[1] + h.shape[1]//2],
                   original))
        if(eps!=0 and (k<eps or k_prev<k)):
            print 'eps break'
            break
        if(N!=0 and not(i<N)):
            print 'N break'
            break
        if N!=0:
            print i
        if eps!=0:
            print k
        i+=1
        gc.collect()
        #print k
    err=np.array(err)
    plt.figure()
    plt.plot(np.arange(err.size), err)
    plt.xlabel('iteration')
    plt.ylabel('dif')
    plt.title('compare f(i) and original image')
    plt.show()
    return f

def lucy_richardson_deconvolution_rgb(g, h, eps):
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    result = np.zeros(g.shape)
    print 'lucy-richardson deconvolution'

    print 'red'
    result[:, :, 0] = lucy_richardson_deconvolution(g_r, h,eps)
    print 'green'
    result[:, :, 1] = lucy_richardson_deconvolution(g_g, h,eps)
    print 'blue'
    result[:, :, 2] = lucy_richardson_deconvolution(g_b, h,eps)
    return result

def temp(t):
    return lucy_richardson_deconvolution(*t)



def lucy_richardson_deconvolution_multythread(g,h,eps):
    pool=Pool(processes=mp.cpu_count())
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    values=[(g_r, h, eps),
            (g_g, h, eps),
            (g_b, h, eps)]
    print 'go multythread'
    temp_result=pool.map(temp, values)
    print 'end multythread'
    gc.collect()
    result = np.zeros(g.shape)
    result[:, :, 0]=temp_result[0]
    result[:, :, 1]=temp_result[1]
    result[:, :, 2]=temp_result[2]
    return result

def lucy_richardson_blind_deconvolution(g, n, m, original):
    #init h
    print 'First error',images.compare_images(g, original)
    plt.imsave(fname='l_r_blind/g.bmp', arr=np.uint8(images.correct_image(g)), cmap='gray')
    #h1=conv.random_psf(3,3)
    # h1[1,:3]=1/3.
    ##h=np.zeros(g.shape, dtype=float)
    #h[255:258, 255:258]=h1
    h = (1. / np.sum(g) ** 2) * conv.correlation2(g, g)
    h/=np.sum(h)
    #h=conv.random_psf(g.shape[0], g.shape[1])
    #h/=np.sum(h)
    #h = conv.correlation2(g, g)
   # print np.sum(h)
    #h/=np.sum(h)
    plt.imsave(fname='l_r_blind/init_h.bmp', arr=np.uint8(images.correct_image(h*255)), cmap='gray')
    print 'h sum=', np.sum(h)
    #h[255:258, 255:258]=h1
    f=np.copy(g)
    print 'blind l-r'
    print '0  %'
    errors=[]
    for i in range(n):
        f_prev=np.copy(f)

        #print 'h:'
        #print '-- 0 %'
        for k in range(m):
            p = g / (conv.convolution2(f, h))
            flr=np.fliplr(np.flipud(f))
            h=conv.convolution2(p,flr)*h
            h/=np.sum(f)
            #h/=np.sum(h)
            #h = (1. / np.sum(f)) * (h * conv.correlation2(f, p))
            # p=g/(sg.convolve2d(f,h, mode='same'))
            # h=(1./np.sum(f))*(h*sg.correlate2d(f,p,mode='same'))
            #print '--', float(k+1) / m * 100, '%'

            # f=(1./np.sum(h))*(f*sg.correlate2d(h,p,mode='same'))
            # print 'f:'
            # print '-- 0 %'
            h /= np.sum(h)
            for k in range(m):
                p = g / (conv.convolution2(f, h))
                hlr = np.fliplr(np.flipud(h))
                f = conv.convolution2(p, hlr) * f
                f/=np.sum(h)
                # f = (1. / np.sum(h)) * (f * conv.correlation2(h, p))
                # p=g/(sg.convolve2d(f,h, mode='same'))
                # print '--', float(k+1) / m * 100, '%'
            #images.check_image(f)
        print (float(i+1) / n) * 100, ' %'
        name='l_r_blind/new_lena'+str(i)+'.bmp'
        h_name='l_r_blind/h_'+str(i)+'.bmp'
        plt.imsave(fname=name, arr=np.uint8(images.correct_image(f)), cmap='gray')
        #print 't and f comp', images.compare_images(temp, f)
        plt.imsave(fname=h_name, arr=np.uint8(images.correct_image(h*255)), cmap='gray')
        images.check_image(images.correct_image(f))
        error = images.compare_images(images.correct_image(f), original)
        errors.append(error)
        print 'err =',error
    errors=np.array(errors)
    plt.figure()
    plt.plot(np.arange(errors.size), errors)
    plt.xlabel('Steps')
    plt.ylabel('Dif btw original and f_k')
    plt.show()
    return f,h


def lucy_richardson_blind_deconvolution0_1(g, n, m, original):
    #init h
    print 'First error',images.compare_images(g, original)
    plt.imsave(fname='l_r_blind/g.bmp', arr=np.uint8(images.make0to255(g)), cmap='gray')
    #h1=conv.random_psf(3,3)
    # h1[1,:3]=1/3.
    ##h=np.zeros(g.shape, dtype=float)
    #h[255:258, 255:258]=h1
    h = (1. / np.sum(g) ** 2) * conv.correlation2(g, g)
    h/=np.sum(h)
    #h=conv.random_psf(g.shape[0], g.shape[1])
    #h/=np.sum(h)
    #h = conv.correlation2(g, g)
   # print np.sum(h)
    #h/=np.sum(h)
    plt.imsave(fname='l_r_blind/init_h.bmp', arr=np.uint8(images.make0to255(h)), cmap='gray')
    print 'h sum=', np.sum(h)
    #h[255:258, 255:258]=h1
    f=np.copy(g)
    print 'blind l-r'
    print '0  %'
    errors=[]
    for i in range(n):
        f_prev=np.copy(f)

        #print 'h:'
        #print '-- 0 %'
        for k in range(m):
            p = g / (conv.convolution2(f, h))
            flr=np.fliplr(np.flipud(f))
            h=conv.convolution2(p,flr)*h
            h/=np.sum(f)
            #h/=np.sum(h)
            #h = (1. / np.sum(f)) * (h * conv.correlation2(f, p))
            # p=g/(sg.convolve2d(f,h, mode='same'))
            # h=(1./np.sum(f))*(h*sg.correlate2d(f,p,mode='same'))
            #print '--', float(k+1) / m * 100, '%'

            # f=(1./np.sum(h))*(f*sg.correlate2d(h,p,mode='same'))
            # print 'f:'
            # print '-- 0 %'
            h /= np.sum(h)
            for k in range(m):
                p = g / (conv.convolution2(f, h))
                hlr = np.fliplr(np.flipud(h))
                f = conv.convolution2(p, hlr) * f
                # f = (1. / np.sum(h)) * (f * conv.correlation2(h, p))
                # p=g/(sg.convolve2d(f,h, mode='same'))
                # print '--', float(k+1) / m * 100, '%'
            #images.check_image(f)
        print (float(i+1) / n) * 100, ' %'
        name='l_r_blind/new_lena'+str(i)+'.bmp'
        h_name='l_r_blind/h_'+str(i)+'.bmp'
        plt.imsave(fname=name, arr=np.uint8(images.make0to255(f)), cmap='gray')
        #print 't and f comp', images.compare_images(temp, f)
        plt.imsave(fname=h_name, arr=np.uint8(images.make0to255(h)), cmap='gray')
        images.check_image(images.correct_image(f))
        error = images.compare_images(images.correct_image(f), original)
        errors.append(error)
        print 'err =',error
    errors=np.array(errors)
    plt.figure()
    plt.plot(np.arange(errors.size), errors)
    plt.xlabel('Steps')
    plt.ylabel('Dif btw original and f_k')
    plt.show()
    return f,h

def lucy_richardson_blind_deconvolution_pir(g, n, m,d, max_psf_size=0, init_h_mode='gaussian', original=None):
    # g - blurred image
    # m, n - count of interation in RL-method
    # max_psf_size - How long must be PSF
    # init_h_mode - How initialize PSF:
    #   'gaussian' - gaussian PSF size = 3x3, sigma =
    #   'horizontal' - motion blur size 3x3, ang=0
    #   'vertical' - motion blir size 3x3 ang = 90
    #   'wow'      - init with g
    plt.imsave(fname='l_r_blind/g.bmp', arr=np.uint8(g), cmap='gray')
    if max_psf_size==0:
        max_psf_size = min(g.shape[0], g.shape[1])

    #
    # init h:
    #
    if init_h_mode=='gaussian':
        h = conv.gaussian(1,3,3)
    elif init_h_mode == 'horizontal':
        h = np.array([[0,0,0],
                   [1,1,1],
                   [0,0,0]], dtype=float)/3.
    elif init_h_mode == 'vertical':
        h  =np.array([[0,1,0],
                   [0,1,0],
                   [0,1,0]], dtype=float)/3.
    elif init_h_mode == 'wow':
        mini_g=smisc.imresize(g, (3,3))
        h = (1. / np.sum(mini_g) ** 2) * conv.correlation2(mini_g, mini_g)
        h /= np.sum(h)
    else:
        h = conv.random_psf(3,3)
    print h
    s=3
    err=[]
    while (s<=max_psf_size):
        print 'size =',s
        mini_g=smisc.imresize(g, (s,s))
        f = lucy_richardson_deconvolution(mini_g, h, 20000)
        if (s!=3):
            h=smisc.imresize(h, (s,s))
        for k in range (d):
            if(k!=0):
                f = lucy_richardson_deconvolution(mini_g, h, 20000)
            for i in range(n):
                #f_prev = np.copy(f)

                #Correct h
                h_prev=np.copy(h)
                for k in range(m):
                    p = mini_g / (conv.convolution2(f, h))
                    flr = np.fliplr(np.flipud(f))
                    h = conv.convolution2(p, flr)*h
                    #h /= np.sum(h)
                    #print 'h =',h

                for k in range(m):
                    p = mini_g / (conv.convolution2(h_prev, f))
                    hlr = np.fliplr(np.flipud(h_prev))
                    f = conv.convolution2(p, hlr) * f
                print (float(i + 1) / n) * 100, ' %'
                name = 'l_r_blind/new_lena' +str(s)+'_'+ str(i) + '.bmp'
                h_name = 'l_r_blind/h_'+str(s)+'_' + str(i) + '.bmp'
                plt.imsave(fname=name, arr=np.uint8(images.correct_image(f)), cmap='gray')
                plt.imsave(fname=h_name, arr=np.uint8(images.correct_image(h*255)), cmap='gray')
        s=int(s*math.sqrt(2))
        #
        if(original!=None):
            good_f = lucy_richardson_deconvolution(g, h, 20000)
            print good_f.shape
            dx = (good_f.shape[0]-original.shape[0])//2
            dy = (good_f.shape[1]-original.shape[1])//2
            err.append(images.compare_images(good_f[dx:original.shape[0]+dx,
                                                    dy:original.shape[1] + dy],
                                             original))
    err=np.array(err)
    plt.figure()
    plt.plot(np.arange(err.size), err)
    #plt.imshow(h,cmap='gray')
    plt.show()
    f = lucy_richardson_deconvolution(g, h, 5000)
    return f, h
