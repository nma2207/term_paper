import numpy as np
import random
import math
from PIL import Image, ImageDraw, ImageEnhance


def step_filter(gamma):
    image=Image.open("work.jpg")
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
    image.save("step_filter/res2.jpg", "JPEG")
    del draw

def median_filtr():
    image=Image.open("test.jpg")
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
    image.save("median_filter/res1.jpg", "JPEG")
    del draw

def minus():
    image1=Image.open("test.jpg")
    draw1=ImageDraw.Draw(image1)
    image2=Image.open("median_filter/res1.jpg")
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
    image=Image.open("olen.jpg")
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
    image.save("laplassian/res1.jpg", "JPEG")
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
    image.save("gauss_filter/res1.jpg")

def increase_in_clearness():
    f=[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
    n=3
    image=Image.open("olen.jpg")
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
    image.save("increase_in_clearness/res1.jpg")

def box_filter():
    f=[[1./9,1./9,1./9],[1./9,1./9,1./9],[1./9,1./9,1./9]]
    n=3
    image=Image.open("olen.jpg")
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
    image.save("box_filter/res1.jpg")

def add_alfa(alfa):
    image=Image.open("olen.jpg")
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
    image.save("add_alfa/res1.jpg")

def increase_in_sharpness():
    #povishenie rezkosti
    f1=[[0,0,0],[0,2,0],[0,0,0]]
    f2=[[1./9,1./9,1./9],[1./9,1./9,1./9],[1./9,1./9,1./9]]
    n=3
    image=Image.open("olen.jpg")
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
    image.save("increase_in_sharpness/res1.jpg")
    Image._show(image)

def addapt_loc_filter():
    image=Image.open("test.jpg")
    draw=ImageDraw.Draw(image)
    width=image.size[0]
    height=image.size[1]
    pix=image.load()

    q=[]
    for i in range(width):
        for j in range(height):
            q.append(pix[i,j])
    gl_disp = disp(q, width * height)
    print gl_disp
    n=3
    for i in range(n//2, width-n//2):
        for j in range(n//2, height-n//2):
            #print i-n//2,i+n//2+1,j-n//2,j+n//2+1
            #d=disp(pix[i-n//2:i+n//2+1,j-n//2:j+n//2+1],n,n)
            r=[]
            for p in range(-(n - 1) // 2, (n + 1) // 2):
                for q in range(-(n - 1) // 2, (n + 1) // 2):
                    r.append(pix[i+p,j+q])
            d=disp(r,n*n)
            m_l=mean(r,n*n)
            res=[.0,.0,.0]
            for p in range(3):
                k=0.0
                if(d[p]<gl_disp[p]):
                    k=1
                else:
                    k=d[p]/gl_disp[p]
                res[p]=int(pix[i,j][p]-k*(pix[i,j][p]-m_l[p]))
            draw.point((i,j),(res[0],res[1],res[2]))
    image.save("addapt_loc_filter/res2.jpg")




def mean(pix, n):
    a=[.0,.0,.0]
    for i in range(n):
        for p in range(3):
            a[p]+=pix[i][p]

    a[0]/=n
    a[1] /= n
    a[2] /= n
    return a


def disp(pix,n):
    res=[.0,.0,.0]
    m=mean(pix,n)
    for i in range(n):
        for p in range(3):
            res[p]+=(m[p]-pix[i][p])**2
    for p in range(3):
        res[p]/=n;
    return res


def main():
    #laplassian()
    #median_filtr()
    #add_alfa(0.5);
    #minus()
    #box_filter()
    #step_filter(0.7)
    #gauss_filter(1.5);
    #increase_in_clearness()
    #increase_in_sharpness()
    addapt_loc_filter()

if __name__ == "__main__":
    main()