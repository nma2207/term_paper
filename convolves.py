# coding=utf-8
from scipy import signal as sc_s
import numpy as np
import math

def convolution(f,h):
    result=sc_s.convolve2d(f,h)
    return result

def convolution2(f,h):
    height=f.shape[0]*2
    width=f.shape[1]*2
    f1=np.zeros((height,width))
    h1=np.zeros((height,width))
    f1[0:f.shape[0], 0:f.shape[1]]=f
    h1[0:h.shape[0], 0:h.shape[1]] = h
    F=np.fft.fft2(f1)
    H=np.fft.fft2(h1)
    G=F*H
    g1=np.fft.ifft2(G)
    g1=np.real(g1)
    g=g1[h.shape[0]//2:f.shape[0]+h.shape[0]//2, h.shape[1]//2:f.shape[1]+h.shape[1]//2]

    return g

def correlation2(f,h):
    height=f.shape[0]*2
    width=f.shape[1]*2
    f1=np.zeros((height,width))
    h1=np.zeros((height,width))
    f1[0:f.shape[0], 0:f.shape[1]]=f
    h1[0:h.shape[0], 0:h.shape[1]] = h
    F=np.fft.fft2(f1)
    H=np.fft.fft2(h1)

    F=np.conjugate(F)
    G=F*H
    #почему
    G=np.conjugate(G)
    g1=np.fft.ifft2(G)
    g1=np.real(g1)
    g=np.zeros((f.shape[0], f.shape[1]), dtype=float)
    #правый нижний угол
    g[:h.shape[0]//2, :h.shape[1]//2]=g1[g1.shape[0]-h.shape[0]//2:g1.shape[0], g1.shape[1]-h.shape[1]//2:g1.shape[1]]
    #левый верхний
    g[h.shape[0]//2:g.shape[0],  h.shape[1]//2:g.shape[1]]=g1[:f.shape[0]-h.shape[0]//2, :f.shape[1]-h.shape[1]//2]
    #левый нижний верхний
    g[h.shape[0]//2:g.shape[0], :h.shape[1]//2]=g1[:f.shape[0]-h.shape[0]//2, g1.shape[1]-h.shape[1]//2:g1.shape[1]]
    #праывй верхний
    g[:h.shape[0]//2, h.shape[1]//2:g.shape[1]] = g1[g1.shape[0]-h.shape[0]//2:g1.shape[0], :f.shape[1]-h.shape[1]//2]
    #g = g1[h.shape[0] // 2:f.shape[0] + h.shape[0] // 2, h.shape[1] // 2:f.shape[1] + h.shape[1] // 2]
    return g

def convolution_rgb(f, h):

    f_r=f[:,:,0]
    f_g=f[:,:,1]
    f_b=f[:,:,2]

    result0 = convolution2(f_r, h)
    result1 = convolution2(f_g, h)
    result2 = convolution2(f_b, h)

    result = np.zeros((result0.shape[0], result0.shape[1],3))
    result[:, :, 0] = result0
    result[:, :, 1] = result1
    result[:, :, 2] = result2
    return result

def gauss(x,y,sigma):
    twoPi = math.pi * 2
    return (1/(twoPi*sigma*sigma))*math.exp(-(x*x+y*y)/float(2*sigma*sigma))

def gaussian(sigma,n,m):
    f=np.array([[gauss(i,j,sigma) for j in range (-(m-1)//2, (m+1)//2)] for i in range(-(n-1)//2, (n+1)//2)])
    f = f / np.sum(f)
    return f

def add_normal_noise(f, mean=0, sigma=1):
    noise = np.random.normal(mean, sigma, (f.shape[0], f.shape[1]))
    f+=noise
    return f,noise

def add_normal_noise_rgb(f, mean=0, sigma=1):
    f_r=f[:,:,0]
    f_g=f[:,:,1]
    f_b=f[:,:,2]
    f_r,noise_r = add_normal_noise(f_r, mean, sigma)
    f_g,noise_g = add_normal_noise(f_g, mean, sigma)
    f_b,noise_b = add_normal_noise(f_b, mean, sigma)
    result = np.zeros((noise_r.shape[0], noise_r.shape[1],3))
    result[:, :, 0] = noise_r
    result[:, :, 1] = noise_g
    result[:, :, 2] = noise_b
    f[:, :, 0] = f_r
    f[:, :, 1] = f_g
    f[:, :, 2] = f_b

    return f,result


#ang can be only 0, 45 or 90
# def motion_blur(len, ang):
#     result=np.zeros((len, len), dtype=float)
#     if(ang==0):
#         for i in range(0, len):
#             result[len//2, i]=1.0
#     elif (ang==45):
#         for i in range(0, len):
#             result[len-i-1, i]=1.0
#     elif (ang==90):
#         for i in range(0, len):
#             result[i, len//2]=1.0
#     result/=np.sum(result)
#     return result
def find(x,y):
    result=[]
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if((x[i]==y[j]).prod()==1):
                result.append(x[i])
    return np.array(result)

def motion_blur(len, ang):
    eps=2.220446049250313e-016
    len=max(1, len)
    half=(len-1)/2
    phi=(ang%180)/180.*math.pi
    cosphi=math.cos(phi)
    sinphi=math.sin(phi)
    xsign=np.sign(cosphi)
    linewdth=1
    sx=int(half*cosphi+linewdth*xsign-len*eps)
    sy=int(half*sinphi+linewdth-len*eps)
    xx=np.arange(0,sx+xsign, xsign)
    yy=np.arange(0, sy+1)
    x,y=np.meshgrid(xx,yy)
    dist2line=(y*cosphi-x*sinphi)
    rad=np.sqrt(x*x+y*y)
    indexs1=np.where(rad>=half)
    indexs2=np.where(np.fabs(dist2line)<=linewdth)
    indexs1=np.array(indexs1)
    indexs1=indexs1.transpose()
    indexs2=np.array(indexs2)
    indexs2=indexs2.transpose()
    indexs=find(indexs1,indexs2)
    x2lastpix=half-np.fabs((x[indexs[:,0], indexs[:,1]]+dist2line[indexs[:,0], indexs[:,1]]*sinphi)/cosphi)
    dist2line[indexs[:,0], indexs[:,1]]=np.sqrt(dist2line[indexs[:,0], indexs[:,1]]**2+x2lastpix**2)
    dist2line=linewdth+eps-abs(dist2line)
    dist2line[np.where(dist2line < 0)] = 0
    h=np.rot90(dist2line,2)
    old_h=np.copy(h)
    h_w=h.shape[0]
    h_h=h.shape[1]
    d_w=dist2line.shape[0]
    d_h=dist2line.shape[1]
    h=np.zeros((h_w+d_w-1, h_h+d_h-1), dtype=np.float64)
    h[0:h_w, 0:h_h]=old_h
    h[h_w-1: h_w+d_w-1, h_h-1: h_h+d_h-1]=dist2line
    h=h/(np.sum(h)+eps*len*len)
    #h(end + (1:end)-1, end + (1:end)-1) = dist2line;
    if cosphi>0:
        h=np.flipud(h)
    return h



def sobel_filter_X():
    return np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]], dtype=float
                    )
def sobel_filter_Y():
    return np.array([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]], dtype=float)

def averaging_filter(n,m):
    result=np.ones((n,m), dtype=float)
    result/=(n*m)
    return result

def random_psf(n,m):
    result=np.random.rand(n,m)
    result/=np.sum(result)
    return result
# If I need code below, I'll restore they

# def step_filter(gamma):
#     image=Image.open("kosmichi.jpg")
#     draw=ImageDraw.Draw(image)
#     width=image.size[0]
#     height=image.size[1]
#     pix=image.load()
#     for i in range(width):
#         for j in range(height):
#             a = int(((pix[i,j][0] / 255.) ** gamma)*255)
#             b = int(((pix[i,j][1] / 255.) ** gamma)*255)
#             c = int(((pix[i,j][2] / 255.) ** gamma)*255)
#             draw.point((i,j),(a,b,c))
#     image.save("step_filter/res3.jpg", "JPEG")
#     del draw
#
# def median_filtr():
#     image=Image.open("kosmichi.jpg")
#     corn=[[0,1,0],[1,1,1],[0,1,0]]
#     draw=ImageDraw.Draw(image)
#     width=image.size[0]
#     height=image.size[1]
#     pix=image.load()
#     n=3
#     for i in range(n/2,width-n/2):
#         for j in range(n/2,height-n/2):
#             f = []
#             for p in range(n):
#                 for q in range(n):
#                     if corn[p][q]==1:
#                         f.append(pix[i+p-n/2, j+q-n/2])
#             #f=[[pix[l,m] for l in range(j,j+n)] for m in range (i,i+n)]
#             f.sort()
#             #print len(f)
#             draw.point((i,j),(f[(len(f)+1)/2]))
#     image.save("median_filter/res2.jpg", "JPEG")
#     del draw
#
# def minus():
#     image1=Image.open("olen_wb.jpg")
#     draw1=ImageDraw.Draw(image1)
#     image2=Image.open("addapt_loc_filter/res4.jpg")
#     draw2=ImageDraw.ImageDraw(image2)
#     width=image1.size[0]
#     height=image1.size[1]
#     pix1=image1.load()
#     pix2=image2.load()
#     for i in range(width):
#         for j in range (height):
#             draw1.point((i, j), (pix1[i,j][0]-pix2[i,j][0],pix1[i,j][1]-pix2[i,j][1],pix1[i,j][2]-pix2[i,j][2]))
#     image1.save("res34.jpg", "JPEG")
#     del draw1
#     del draw2
#
# def laplassian():
#     image=Image.open("kosmichi.jpg")
#     draw=ImageDraw.Draw(image)
#     width=image.size[0]
#     height=image.size[1]
#     pix=image.load()
#     f=[[0.,-1.,0. ],[-1.,4.,-1. ],[0.,-1.,0. ]]
#     n=3
#     d=[[[0 for i in range(width)] for j in range (height)],[[0 for i in range(width)] for j in range(height)],[[0 for i in range(width)] for j in range(height)]]
#     for k in range(3):
#         for i in range (1, width-1):
#             for j in range(1, height-1):
#                 d[k][j][i]=4*pix[i,j][k]-(pix[i+1,j][k]+pix[i-1,j][k]+pix[i,j+1][k]+pix[i,j-1][k])#+pix[i+1,j+1][k]+pix[i+1,j-1][k]+pix[i-1,j+1][k]+pix[i-1,j-1][k])
#
#
#     print f
#     for i in range(1,width-1):
#         for j in range(1,height-1):
#             a=pix[i,j][0]
#             b=pix[i,j][1]
#             c=pix[i,j][2]
#             r=[0.0,0.0,0.0]
#             #for k in range(3):
#             #   for p in range(-(n-1)/2, (n-1)/2+1):
#             #       for q in range(-(n-1)/2, (n-1)/2+1):
#             #           r[k]+=(f[p+n/2][q+n/2]*pix[i+p,j+q][k])
#             draw.point((i,j),(pix[i,j][0]+d[0][j][i],pix[i,j][1]+d[1][j][i],pix[i,j][2]+d[2][j][i]))
#     image.save("laplassian/res2.jpg", "JPEG")
#     del draw
#

#

#
# def increase_in_clearness():
#     f=[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
#     n=3
#     image=Image.open("kosmichi.jpg")
#     draw=ImageDraw.Draw(image)
#     width=image.size[0]
#     height=image.size[1]
#     pix=image.load()
#     for i in range(n/2,width-n/2,n):
#         for j in range(n/2,height-n/2,n):
#             r=[0,0,0]
#             for k in range(3):
#                 for p in range(-(n-1)//2, (n+1)//2):
#                     for q in range (-(n-1)//2, (n+1)//2):
#                         r[k]+=f[n//2+p][q+n//2]*pix[i+p,j+q][k]
#             draw.point((i,j),(int(r[0]),int(r[1]),int(r[2])))
#     image.save("increase_in_clearness/res2.jpg")
#
# def box_filter():
#     f=[[1./9,1./9,1./9],[1./9,1./9,1./9],[1./9,1./9,1./9]]
#     n=3
#     image=Image.open("kosmichi.jpg")
#     draw=ImageDraw.Draw(image)
#     width=image.size[0]
#     height=image.size[1]
#     pix=image.load()
#     for i in range(n/2,width-n/2,n):
#         for j in range(n/2,height-n/2,n):
#             r=[0,0,0]
#             for k in range(3):
#                 for p in range(-(n-1)//2, (n+1)//2):
#                     for q in range (-(n-1)//2, (n+1)//2):
#                         r[k]+=f[n//2+p][q+n//2]*pix[i+p,j+q][k]
#                         #print r
#             draw.point((i,j),(int(r[0]),int(r[1]),int(r[2])))
#     image.save("box_filter/res2.jpg")
#
# def add_alfa(alfa):
#     image=Image.open("kosmichi.jpg")
#     draw=ImageDraw.Draw(image)
#     width=image.size[0]
#     height=image.size[1]
#     pix=image.load()
#     for i in range(0,width):
#         for j in range(height):
#             a=pix[i,j][0]*alfa
#             b=pix[i,j][1]*alfa
#             c=pix[i,j][2]*alfa
#             draw.point((i,j),(int(a),int(b),int(c)))
#     image.save("add_alfa/res2.jpg")
#
# def increase_in_sharpness():
#     #povishenie rezkosti
#     f1=[[0,0,0],[0,2,0],[0,0,0]]
#     f2=[[1./9,1./9,1./9],[1./9,1./9,1./9],[1./9,1./9,1./9]]
#     n=3
#     image=Image.open("kosmichi.jpg")
#     draw=ImageDraw.Draw(image)
#     width=image.size[0]
#     height=image.size[1]
#     pix=image.load()
#     for i in range(n/2,width-n/2,n):
#         for j in range(n/2,height-n/2,n):
#             r=[0,0,0]
#             for k in range(3):
#                 for p in range(-(n-1)//2, (n+1)//2):
#                     for q in range (-(n-1)//2, (n+1)//2):
#                         r[k]+=(f1[n//2+p][q+n//2]-f2[n//2+p][q+n//2])*pix[i+p,j+q][k]
#                         #print r
#             draw.point((i,j),(int(r[0]),int(r[1]),int(r[2])))
#     image.save("increase_in_sharpness/res2.jpg")
#     Image._show(image)
#
#
#
# def addapt_loc_filter():
#     im=plb.imread("kosmichi.jpg")
#     bwi=make_black_white_im(im)
#     n=7
#     h=len(im[:])
#     w=len(im[0,:])
#     bw=np.zeros((h,w,3))
#     bw[:,:,0]=bwi
#     bw[:,:,1]=bwi
#     bw[:,:,2]=bwi
#     plb.imsave("kosmichi_wb.jpg",bw)
#     d_gl=np.var(bwi)
#     for i in range(n/2,h-n/2):
#         for j in range(n/2,w-n/2):
#             m=np.mean(bwi[i-n//2:i+n//2+1,j-n//2:j+n//2+1])
#             d=np.var(bwi[i-n//2:i+n//2+1,j-n//2:j+n//2+1])
#             k=0
#             if (d_gl>d):
#                 k = 1
#             else:
#                 k = float(d_gl)/d
#             bw[i,j,0]=bwi[i,j]-k*(bwi[i,j]-m)
#             bw[i, j, 1] = bwi[i, j] - k * (bwi[i, j] - m)
#             bw[i, j, 2] = bwi[i, j] - k * (bwi[i, j] - m)
#     bw = np.uint8(bw)
#     plb.imsave("addapt_loc_filter/res9.jpg",bw)