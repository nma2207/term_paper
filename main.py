import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import random
import math
from PIL import Image, ImageDraw, ImageEnhance
import scipy
from scipy import signal
from scipy import ndimage
def step_filter(gamma):
    image=Image.open("kosmichi.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    for i in range(width):
        for j in range(height):
            a = int(((pix[i,j][0] / 255.) ** gamma)*255)
            b = int(((pix[i,j][1] / 255.) ** gamma)*255)
            c = int(((pix[i,j][2] / 255.) ** gamma)*255)
            draw.point((i,j),(a,b,c))
    image.save("step_filter/res3.jpg", "JPEG")
    del draw

def median_filtr():
    image=Image.open("kosmichi.jpg")
    corn=[[0,1,0],[1,1,1],[0,1,0]]
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    n=3
    for i in range(n/2,width-n/2):
        for j in range(n/2,height-n/2):
            f = []
            for p in range(n):
                for q in range(n):
                    if corn[p][q]==1:
                        f.append(pix[i+p-n/2, j+q-n/2])
            #f=[[pix[l,m] for l in range(j,j+n)] for m in range (i,i+n)]
            f.sort()
            #print len(f)
            draw.point((i,j),(f[(len(f)+1)/2]))
    image.save("median_filter/res2.jpg", "JPEG")
    del draw

def minus():
    image1=Image.open("olen_wb.jpg")
    draw1=ImageDraw.Draw(image1)
    image2=Image.open("addapt_loc_filter/res4.jpg")
    draw2=ImageDraw.ImageDraw(image2)
    width=image1.size[0]
    height=image1.size[1]
    pix1=image1.load()
    pix2=image2.load()
    for i in range(width):
        for j in range (height):
            draw1.point((i, j), (pix1[i,j][0]-pix2[i,j][0],pix1[i,j][1]-pix2[i,j][1],pix1[i,j][2]-pix2[i,j][2]))
    image1.save("res34.jpg", "JPEG")
    del draw1
    del draw2

def laplassian():
    image=Image.open("kosmichi.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    f=[[0.,-1.,0. ],[-1.,4.,-1. ],[0.,-1.,0. ]]
    n=3
    d=[[[0 for i in range(width)] for j in range (height)],[[0 for i in range(width)] for j in range(height)],[[0 for i in range(width)] for j in range(height)]]
    for k in range(3):
        for i in range (1, width-1):
            for j in range(1, height-1):
                d[k][j][i]=4*pix[i,j][k]-(pix[i+1,j][k]+pix[i-1,j][k]+pix[i,j+1][k]+pix[i,j-1][k])#+pix[i+1,j+1][k]+pix[i+1,j-1][k]+pix[i-1,j+1][k]+pix[i-1,j-1][k])


    print f
    for i in range(1,width-1):
        for j in range(1,height-1):
            a=pix[i,j][0]
            b=pix[i,j][1]
            c=pix[i,j][2]
            r=[0.0,0.0,0.0]
            #for k in range(3):
            #   for p in range(-(n-1)/2, (n-1)/2+1):
            #       for q in range(-(n-1)/2, (n-1)/2+1):
            #           r[k]+=(f[p+n/2][q+n/2]*pix[i+p,j+q][k])
            draw.point((i,j),(pix[i,j][0]+d[0][j][i],pix[i,j][1]+d[1][j][i],pix[i,j][2]+d[2][j][i]))
    image.save("laplassian/res2.jpg", "JPEG")
    del draw

def gauss(x,y,sigma):
    twoPi = math.pi * 2
    return (1/math.sqrt(twoPi)*sigma)*math.exp(-(x*x+y*y)/float(2*sigma*sigma))

def gauss_filter(sigma,n,m):
    #n=3
    f=np.array([[gauss(i,j,sigma) for j in range (-(m-1)//2, (m+1)//2)] for i in range(-(n-1)//2, (n+1)//2)])
    f = f / np.sum(f)
    print 'go'
    #image=Image.open("kosmichi.jpg")
    #draw=ImageDraw.Draw(image)
    #width=image.size[0]
   # height=image.size[1]
    #pix=image.load()
    #for i in range(n/2,width-n/2):
     #   for j in range(n/2,height-n/2):
     #       r=[0,0,0]
     #       for k in range(3):
      #          for p in range(-(n-1)//2, (n+1)//2):
      #              for q in range (-(n-1)//2, (n+1)//2):
     #                   r[k]+=f[n//2+p][q+n//2]*pix[i+p,j+q][k]
      #      draw.point((i,j),(int(r[0]),int(r[1]),int(r[2])))
    #image.save("gauss_filter/res2.jpg")
    return f

def increase_in_clearness():
    f=[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
    n=3
    image=Image.open("kosmichi.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    for i in range(n/2,width-n/2,n):
        for j in range(n/2,height-n/2,n):
            r=[0,0,0]
            for k in range(3):
                for p in range(-(n-1)//2, (n+1)//2):
                    for q in range (-(n-1)//2, (n+1)//2):
                        r[k]+=f[n//2+p][q+n//2]*pix[i+p,j+q][k]
            draw.point((i,j),(int(r[0]),int(r[1]),int(r[2])))
    image.save("increase_in_clearness/res2.jpg")

def box_filter():
    f=[[1./9,1./9,1./9],[1./9,1./9,1./9],[1./9,1./9,1./9]]
    n=3
    image=Image.open("kosmichi.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    for i in range(n/2,width-n/2,n):
        for j in range(n/2,height-n/2,n):
            r=[0,0,0]
            for k in range(3):
                for p in range(-(n-1)//2, (n+1)//2):
                    for q in range (-(n-1)//2, (n+1)//2):
                        r[k]+=f[n//2+p][q+n//2]*pix[i+p,j+q][k]
                        #print r
            draw.point((i,j),(int(r[0]),int(r[1]),int(r[2])))
    image.save("box_filter/res2.jpg")

def add_alfa(alfa):
    image=Image.open("kosmichi.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    for i in range(0,width):
        for j in range(height):
            a=pix[i,j][0]*alfa
            b=pix[i,j][1]*alfa
            c=pix[i,j][2]*alfa
            draw.point((i,j),(int(a),int(b),int(c)))
    image.save("add_alfa/res2.jpg")

def increase_in_sharpness():
    #povishenie rezkosti
    f1=[[0,0,0],[0,2,0],[0,0,0]]
    f2=[[1./9,1./9,1./9],[1./9,1./9,1./9],[1./9,1./9,1./9]]
    n=3
    image=Image.open("kosmichi.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    for i in range(n/2,width-n/2,n):
        for j in range(n/2,height-n/2,n):
            r=[0,0,0]
            for k in range(3):
                for p in range(-(n-1)//2, (n+1)//2):
                    for q in range (-(n-1)//2, (n+1)//2):
                        r[k]+=(f1[n//2+p][q+n//2]-f2[n//2+p][q+n//2])*pix[i+p,j+q][k]
                        #print r
            draw.point((i,j),(int(r[0]),int(r[1]),int(r[2])))
    image.save("increase_in_sharpness/res2.jpg")
    Image._show(image)

def make_black_white_im(pix):
    #im1=(pix[:,:,0])/3
    #im2=(pix[:,:,1])/3
    #im3=(pix[:,:,2])/3
    #res=im1+im2+im3
    h=pix.shape[0]
    w=pix.shape[1]
    res=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            res[i,j]=np.sum(pix[i,j])
    res=res//3

    return res

def addapt_loc_filter():
    im=plb.imread("kosmichi.jpg")
    bwi=make_black_white_im(im)
    n=7
    h=len(im[:])
    w=len(im[0,:])
    bw=np.zeros((h,w,3))
    bw[:,:,0]=bwi
    bw[:,:,1]=bwi
    bw[:,:,2]=bwi
    plb.imsave("kosmichi_wb.jpg",bw)
    d_gl=np.var(bwi)
    for i in range(n/2,h-n/2):
        for j in range(n/2,w-n/2):
            m=np.mean(bwi[i-n//2:i+n//2+1,j-n//2:j+n//2+1])
            d=np.var(bwi[i-n//2:i+n//2+1,j-n//2:j+n//2+1])
            k=0
            if (d_gl>d):
                k = 1
            else:
                k = float(d_gl)/d
            bw[i,j,0]=bwi[i,j]-k*(bwi[i,j]-m)
            bw[i, j, 1] = bwi[i, j] - k * (bwi[i, j] - m)
            bw[i, j, 2] = bwi[i, j] - k * (bwi[i, j] - m)
    bw = np.uint8(bw)
    plb.imsave("addapt_loc_filter/res9.jpg",bw)
def convolution(f,h,n):
    #print 'fft',np.fft.fft2(h)
    #result=scipy.ndimage.filters.convolve(f,h)
    # f1=np.zeros((f.shape[0]+h.shape[0],f.shape[1]+h.shape[1] ))
    # f1[h.shape[0]//2:h.shape[0]//2+f.shape[0],h.shape[1]//2:h.shape[1]//2+f.shape[1]]=f
    result=scipy.signal.convolve2d(f,h)
    # result=result[h.shape[0]//2:h.shape[0]//2+f.shape[0],h.shape[1]//2:h.shape[1]//2+f.shape[1]]
    #result=result%255
    return result
    #return g
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


def wiener_filter(g, h,k):
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
    k1=np.zeros((2*width_g, 2*height_g))
    k1[0:k.shape[0],0:k.shape[1]]=k
    K=np.fft.fft2(k1)
    #F=(np.abs(H)**2/(H*((np.abs(H)**2))+K))*G
    F = (np.abs(H) ** 2 / (H * ((np.abs(H) ** 2)) )) * G
    f=np.fft.ifft2(F)
    f=np.real(f)
    f=f[0:width_g,0:height_g]
    return f




def main():

    im = plb.imread("lena.bmp")

    bw=make_black_white_im(im)
    print bw
    h = gauss_filter(5, 11, 11)
    print 'h=',h
    con = convolution(bw, h, np.array([[0, 0], [0, 0]]))
    print 'con'
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(bw, cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(con, cmap='gray')
    # plt.show()
    print con
    print 'go'
    filt=inverse_filter(con,h)
    #filt=wiener_filter(con,h, np.array([[1,1],[1,1]]))
    bwi=np.zeros((con.shape[0],con.shape[1],3))
    plt.imsave("convolution/res6.jpg", bwi)
    print 'filt'

    plt.figure()
    plt.subplot(1, 3, 1)



    plt.imshow(bw, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(con, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(filt, cmap='gray')
    plt.show()
    plt.imsave("inverse_filter/res1.jpg", bwi)
if __name__ == "__main__":
    main()