import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageEnhance

def sort(f,n):
    for k in range(2,-1,-1):
        for i in range (n*n):
            for j in range (n*n-1):
                #print j,k
                if(f[j][k]>f[j+1][k]):
                    f[j],f[j+1]=f[j+1],f[j]
                    #for p in range(3):
                     #   #f[j][k],f[j+1][k]=f[j+1][k],f[j][k]
                     #   t=f[j][p]
                     #   f[j][p]=f[j+1][p]
                     #   f[j+1][p]=t

def step_filter(gamma):
    image=Image.open("olen.jpg")
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
    image.save("res48.jpg", "JPEG")
    del draw

def median_filtr():
    image=Image.open("olen.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    n=2
    for i in range(0,width-n):
        for j in range(0,height-n):
            f = []
            for p in range(n):
                for q in range(n):
                    f.append(pix[i+p,j+q])
            #f=[[pix[l,m] for l in range(j,j+n)] for m in range (i,i+n)]
            sort(f,n);
            for p in range(n):
                for q in range(n):
                    draw.point((i+p,j+q),(f[(n*n+1)/2][0],f[(n*n+1)/2][1],f[(n*n+1)/2][2]))
    image.save("res47.jpg", "JPEG")
    del draw

def minus():
    image1=Image.open("exp.jpg")
    draw1=ImageDraw.Draw(image1)
    image2=Image.open("res51.jpg")
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
    image=Image.open("LLlPF3XNF4k.jpg")
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
    image.save("res31.jpg", "JPEG")
    del draw
def gauss(x,y,sigma):
    twoPi = math.pi * 2
    return (1/(twoPi*sigma*sigma)*math.exp(-(x*x+y*y)/float(2*sigma*sigma)))
def gauss_filter(sigma):
    n=3
    f=[[gauss(i,j,sigma) for j in range (-(n-1)//2, (n+1)//2)] for i in range(-(n-1)//2, (n+1)//2)]
    image=Image.open("olen.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()
    for i in range(n/2,width-n/2):
        for j in range(n/2,height-n/2):
            r=[0,0,0]
            for k in range(3):
                for p in range(-(n-1)//2, (n+1)//2):
                    for q in range (-(n-1)//2, (n+1)//2):
                        r[k]+=f[n//2+p][q+n//2]*pix[i+p,j+q][k]
            draw.point((i,j),(int(r[0]),int(r[1]),int(r[2])))
    image.save("res49.jpg")
def increase_in_clearness():
    f=[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
    n=3
    image=Image.open("exp.jpg")
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
    image.save("res51.jpg")
def main():
    #laplassian()
    #median_filtr()
    minus()
    #step_filter(1.2)
    #gauss_filter(1);
    #increase_in_clearness()

if __name__ == "__main__":
    main()

