import numpy as np
import convolves as conv
import images
import multiprocessing as mp
from multiprocessing import Pool
from matplotlib import  pyplot as plt
import scipy.signal as sg

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



def wiener_filter(g, h,n, original):
    width_g=g.shape[0]
    height_g=g.shape[1]
    width_h=h.shape[0]
    height_h=h.shape[1]
    g1=np.zeros((2*width_g,2*height_g))
    h1=np.zeros((2*width_g, 2*height_g))
    n1=np.zeros((2*width_g, 2*height_g))
    original1=np.zeros((2*width_g, 2*height_g))
    g1[0:width_g,0:height_g]=g
    h1[0:width_h, 0:height_h] = h
    n1[0:n.shape[0],0:n.shape[1]]=n
    original1[0:original.shape[0], 0:original.shape[1]] = original
    N=np.fft.fft2(n1)
    ORIGINAL=np.fft.fft2(original1)
    K=(np.abs(N)**2)/(np.abs(ORIGINAL)**2)
    G=np.fft.fft2(g1)
    H=np.fft.fft2(h1)
    F = (np.abs(H) ** 2 / (H * ((np.abs(H) ** 2)+K) )) * G
    f=np.fft.ifft2(F)
    f=np.real(f)
    f=f[0:width_g,0:height_g]
    return f

def wiener_filter_rgb(g,h,n,original):
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    n_r=n[:,:,0]
    n_g=n[:,:,1]
    n_b=n[:,:,2]
    original_r=original[:,:,0]
    original_g=original[:,:,1]
    original_b=original[:,:,2]
    result=np.zeros(g.shape)
    result[:, :, 0] =  wiener_filter(g_r, h,n_r, original_r)
    result[:, :, 1] =  wiener_filter(g_g, h,n_g, original_g)
    result[:, :, 2] =  wiener_filter(g_b, h,n_b, original_b)
    return result

def tickhonov_regularization(g,h):
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
    p1[0:3,0:3]=p
    P=np.fft.fft2(p1)
    gamma=0.
    F =(np.conjugate(H)/(np.abs(H)**2+gamma*np.abs(P)**2))*G
    f = np.fft.ifft2(F)
    f = np.real(f)
    f = f[0:width_g, 0:height_g]
    return f

def tickhonov_regularization_rgb(g,h):
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    result=np.zeros(g.shape)
    result[:, :, 0] = tickhonov_regularization(g_r, h)
    result[:, :, 1] = tickhonov_regularization(g_g, h)
    result[:, :, 2] = tickhonov_regularization(g_b, h)
    return result


#Filippov's article
def quick_blind_deconvolution(g):
    print 'TODO!!'


def lucy_richardson_devonvolution(g,h,eps): #n - iterations count
    f=g
    f_prev=np.zeros(f.shape, dtype=float)
    k=images.compare_images(f_prev, f)

    while(k>eps):

        f_prev=np.copy(f)
        #print k
        k1=conv.convolution2(f,h)
        #k1=k1[h.shape[0]//2:f.shape[0]+h.shape[0]//2, h.shape[1]//2:f.shape[1]+h.shape[1]//2]
        k2=g/k1
        h1=np.flipud(np.fliplr(h))
        k3=conv.convolution2(k2,h1)
        #k3 = k3[h.shape[0] // 2:f.shape[0] + h.shape[0] // 2, h.shape[1] // 2:f.shape[1] + h.shape[1] // 2]
        f=f*k3
        k=images.compare_images(f_prev, f)
    return f

def lucy_richardson_deconvolution_rgb(g, h, eps):
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    result = np.zeros(g.shape)
    print 'lucy-richardson deconvolution'

    print 'red'
    result[:, :, 0] = lucy_richardson_devonvolution(g_r, h,eps)
    print 'green'
    result[:, :, 1] = lucy_richardson_devonvolution(g_g, h,eps)
    print 'blue'
    result[:, :, 2] = lucy_richardson_devonvolution(g_b, h,eps)
    return result

def temp(t):
    return lucy_richardson_devonvolution(*t)



def lucy_richardson_deconvolution_multythread(g,h,eps):
    pool=Pool(processes=mp.cpu_count())
    g_r=g[:,:,0]
    g_g=g[:,:,1]
    g_b=g[:,:,2]
    values=[(g_r, h, eps),
            (g_g, h, eps),
            (g_b, h, eps)]
    temp_result=pool.map(temp, values)
    result = np.zeros(g.shape)
    result[:, :, 0]=temp_result[0]
    result[:, :, 1]=temp_result[1]
    result[:, :, 2]=temp_result[2]
    return result

def lucy_richardson_blind_deconvolution(g, n, m):
    #init h
    plt.imsave(fname='l_r_blind/g.bmp', arr=np.uint8(images.correct_image(g)), cmap='gray')
    h1=np.zeros((3,3))
    h1[1,:3]=1/3.
    h=np.zeros(g.shape, dtype=float)
    h[255:258, 255:258]=h1
    # h = (1. / np.sum(g) ** 2) * conv.correlation2(g, g)
    #h/=np.sum(h)
    plt.imsave(fname='l_r_blind/init_h.bmp', arr=np.uint8(images.correct_image(h*255)), cmap='gray')
    print 'h sum=', np.sum(h)
    #h[255:258, 255:258]=h1
    f=np.copy(g)
    print 'blind l-r'
    print '0  %'
    for i in range(n):

        #print 'h:'
        #print '-- 0 %'
        for k in range(m):
            p = g / (conv.convolution2(f, h))
            flr=np.fliplr(np.flipud(f))
            h=conv.convolution2(p,flr)*h
            #h = (1. / np.sum(f)) * (h * conv.correlation2(f, p))
            # p=g/(sg.convolve2d(f,h, mode='same'))
            # h=(1./np.sum(f))*(h*sg.correlate2d(f,p,mode='same'))
            #print '--', float(k+1) / m * 100, '%'
        #print 'f:'
        #print '-- 0 %'
        for k in range(m):
            p = g / (conv.convolution2(f, h))
            hlr=np.fliplr(np.flipud(h))
            f=conv.convolution2(p, hlr)*f
            #f = (1. / np.sum(h)) * (f * conv.correlation2(h, p))
            # p=g/(sg.convolve2d(f,h, mode='same'))
            # f=(1./np.sum(h))*(f*sg.correlate2d(h,p,mode='same'))
            #print '--', float(k+1) / m * 100, '%'
        print (float(i+1) / n) * 100, ' %'
        name='l_r_blind/new_lena'+str(i)+'.bmp'
        h_name='l_r_blind/h_'+str(i)+'.bmp'
        plt.imsave(fname=name, arr=np.uint8(images.correct_image(f)), cmap='gray')
        #print 't and f comp', images.compare_images(temp, f)
        plt.imsave(fname=h_name, arr=np.uint8(images.correct_image(h*255)), cmap='gray')
    return f,h
