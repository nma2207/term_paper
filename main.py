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
    return (1/(twoPi*sigma*sigma)*math.exp(-(x*x+y*y)/float(2*sigma*sigma)))

def gauss_filter(sigma,n,m):
    #n=3
    f=np.array([[gauss(i,j,sigma) for j in range (-(m-1)//2, (m+1)//2)] for i in range(-(n-1)//2, (n+1)//2)])
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
    res=255-res
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
    result=scipy.signal.convolve2d(f,h)
    #result=result%255
    return result
    #return g
def inverse_filter(g,h):
    H=np.fft.fft2(h)
    G=np.fft.fft2(g)
    height = G.shape[0]
    width = G.shape[1]
    n = H.shape[0]
    m = H.shape[1]
    F = np.zeros((height, width), dtype=np.complex)
    for i in range(n/2,width-n/2):
        for j in range(m/2,height-m/2):
            F[i - n // 2:i + n // 2+1, j - m // 2:j + m // 2+1] = G[i - n // 2:i + n // 2+1, j - m // 2:j + m // 2+1] /H

    f=np.fft.ifft2(F)
    f=np.real(f)
    return f
def wiener_filter(g, h,k):
    H=np.fft.fft2(h)
    G=np.fft.fft2(g)
    r=(1/H)*(abs(H)**2)/(abs(H)**2+k)
    #What do I do, when H have 0????
    #How to use mult???

    height=G.shape[0]
    width=G.shape[1]
    n=H.shape[0]
    m=H.shape[1]
    F = np.zeros((height, width),dtype=np.complex)
    for i in range(n/2,height-n/2+1):
        for j in range(m/2, width-m/2+1):
            F[i-n//2:i+n//2,j-m//2:j+m//2]=G[i-n//2:i+n//2,j-m//2:j+m//2]*r
            #print G[i-n//2:i+n//2,j-m//2:j+m//2]*r
   # print '\n'
    #print F
   # print ' '
    f=np.fft.ifft2(F)
    #print f
    return f


def main():

    im = plb.imread("lena.bmp")

    bw=make_black_white_im(im)
    #bw=bw/255.0
    print bw
    h = gauss_filter(1, 53, 53)
    #con=scipy.ndimage.gaussian_filter(bw,0.3)
    print 'h=',h
    con = convolution(bw, h, np.array([[1, 1], [1, 1]]))
    print 'con'
    for i in range(con.shape[0]):
        for j in range(con.shape[1]):
            if(con[i,j]>255):
                print 'ERROR'

    con1 = np.int32(con)
    print con
    print 'go'
    filt=inverse_filter(con1,h)
    print filt
    bwi=np.zeros((con.shape[0],con.shape[1],3))
    #con=con*255
    #con1=np.zeros((con.shape[0],con.shape[1]))

    print 'con1=',con1
    bwi[:,:,0]=con1
    bwi[:, :, 1] =con1
    bwi[:, :, 2] = con1
    #print bwi
    #plt.plot(bwi)
    plb.imsave("convolution/res6.jpg", bwi)
    print 'filt'
    for i in range(filt.shape[0]):
        for j in range(filt.shape[1]):
            if(filt[i,j]>255):
                print 'ERROR'
    bwi = np.zeros((filt.shape[0],filt.shape[1], 3))
    #filt=filt*255
    filt1=np.int32(filt)
    bwi[:, :, 0] = filt1
    bwi[:, :, 1] = filt1
    bwi[:, :, 2] = filt1
    print 'filt1=',filt1
    plb.imsave("inverse_filter/res1.jpg", bwi)
if __name__ == "__main__":
    main()