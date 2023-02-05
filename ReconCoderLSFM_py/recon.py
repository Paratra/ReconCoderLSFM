# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:53:26 2017

@author: rl74173
"""
import os
import numpy as np
import tifffile as tf
from pylab import imshow, plot
import psfsim

pi = np.pi
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftn = np.fft.fftn
ifftn = np.fft.ifftn
fftshift = np.fft.fftshift

temppath = 'C:/Users/rl74173/Documents/Python codes/Temp'
join = lambda fn: os.path.join(temppath,fn)

class recon(object):
    
    def __init__(self,img_stack,nph,nangles):
        self.temppath = 'C:/Users/rl74173/Documents/Python codes/Temp'        
        self.img = img_stack
        nz,nx,ny = img_stack.shape
        self.nz = int(nz/nph/nangles)
        self.nx = nx
        self.ny = ny
        self.dx = 0.089
        self.dz = 0.2
        self.nphases = nph
        self.norders = 5
        self.psf = self.getpsf()
        self.phiz = 0.0
        self.spacingz = 0.0
                
    def separate(self, nangle=0):
        z = np.arange(self.nz)
        img_0 = self.img[15*z+(5*nangle+0),:,:]
        img_1 = self.img[15*z+(5*nangle+1),:,:]
        img_2 = self.img[15*z+(5*nangle+2),:,:]
        img_3 = self.img[15*z+(5*nangle+3),:,:]
        img_4 = self.img[15*z+(5*nangle+4),:,:]
        tf.imsave(join('img_0.tif'),fftshift(img_0))
        tf.imsave(join('img_1.tif'),fftshift(img_1))
        tf.imsave(join('img_2.tif'),fftshift(img_2))
        tf.imsave(join('img_3.tif'),fftshift(img_3))
        tf.imsave(join('img_4.tif'),fftshift(img_4))

    def getpsf(self):
        nzh = int(self.nz/2)
        nxh = int(self.nx/2)
        if self.nz%2 == 0:
            lim1 = -(self.dz)*(self.nz)/2
            lim2 = (self.dz)*(self.nz-1) + lim1
        else:
            lim1 = -(self.dz)*(self.nz-1)/2
            lim2 = - lim1
        psf = psfsim.psf()
        psf.setParams(wl=0.51,na=1.20,dx=self.dx,nx=self.nx)      
        psf.getFlatWF()
        psf.get3Dpsf(lim1,lim2,self.dz)
        # use roll
        shift = lambda img: np.roll(np.roll(np.roll(img,nzh,0),nxh,1),nxh,2)
        psf1 = shift(psf.stack)
        return psf1
        
    def findspacingx(self,angle,spacing,Ns=10,r_ang=0.02,r_sp=0.005):
        d_ang = 2*r_ang/Ns
        d_sp = 2*r_sp/Ns
        ang_iter = np.arange(-r_ang,r_ang+d_ang/2,d_ang)+angle
        sp_iter = np.arange(-r_sp,r_sp+d_sp/2,d_sp)+spacing
        s = np.zeros((Ns+1,Ns+1))
        for m,ang in enumerate(ang_iter):
            for n,sp in enumerate(sp_iter):
                print m,n
                self.shift(ang,sp)
                s = self.autocorrelate()
        plot()
        imshow(s,interpolation='nearest')
        # get maximum
        mind = s.argmax()
        angmax = int(mind/(Ns+1))*d_ang - r_ang + angle
        spmax = (mind%(Ns+1))*d_sp - r_sp + spacing
        return angmax,spmax
        
    def autocorrelate(self):
        a = tf.imread(join('imgf_0.tif'))
        b = tf.imread(join('otf_0.tif'))
        c = a*np.conj(b)
        
        a = tf.imread(join('imgfs_0.tif'))
        b = tf.imread(join('otfs_0.tif'))
        d = np.conj(a*np.conj(b))
        
        e = np.sum(c*d) 
        
        return e
    
    def shift(self,angle,spacingx):
        ''' shift 2nd order data '''
        nxh = self.nx
        nyh = self.ny
        nzh = self.nz
        kx = self.dx*np.cos(angle)/spacingx
        ky = self.dx*np.sin(angle)/spacingx
        ysh = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        imgf = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        
        g = lambda p,m,n: np.exp(2j*pi*(kx*m+ky*n)).astype(np.complex64)
        ysh[:,:,:] = np.fromfunction(g,(2*self.nz,2*self.nx,2*self.ny))
        ysh[:,:,:] = np.roll(np.roll(np.roll(ysh,nzh,0),nxh,1),nyh,2)
                
        psfdbl = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        psfdbl[:,:,:] = self.interp(self.psf)
        # 2nd order
        imgf[:,:,:] = fftn(psfdbl)
        tf.imsave(join('otf_0.tif'),imgf)
        imgf[:,:,:] = fftn(psfdbl*ysh)
        tf.imsave(join('otfs_0.tif'),imgf)
                
        # get image data
        a = tf.imread(join('img_0.tif'))
#        b = tf.imread(join('img_1.tif'))
#        c = tf.imread(join('img_2.tif'))
#        d = tf.imread(join('img_3.tif'))
#        e = tf.imread(join('img_4.tif'))
              
        psfdbl = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
        psfdbl[:,:,:] = self.interp(a.astype(np.complex64))
        imgf[:,:,:] = fftn(psfdbl)
        tf.imsave(join('imgf_0.tif'),imgf)
        imgf[:,:,:] = fftn(psfdbl*ysh)
        tf.imsave(join('imgfs_0.tif'),imgf)
#        psfdbl = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
#        psfdbl[:,:,:] = self.interp(b.astype(np.complex64))
#        imgf[:,:,:] = fftn(psfdbl*ysh)
#        tf.imsave(join('imgfs_1.tif'),imgf)
#        psfdbl = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
#        psfdbl[:,:,:] = self.interp(c.astype(np.complex64))
#        imgf[:,:,:] = fftn(psfdbl*ysh)
#        tf.imsave(join('imgfs_2.tif'),imgf)
#        psfdbl = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
#        psfdbl[:,:,:] = self.interp(d.astype(np.complex64))
#        imgf[:,:,:] = fftn(psfdbl*ysh)
#        tf.imsave(join('imgfs_3.tif'),imgf)
#        psfdbl = np.zeros((2*nzh,2*nxh,2*nyh),dtype=np.complex64)
#        psfdbl[:,:,:] = self.interp(e.astype(np.complex64))
#        imgf[:,:,:] = fftn(psfdbl*ysh)
#        tf.imsave(join('imgfs_4.tif'),imgf)
        
    def interp(self,arr):
        nz,nx,ny = arr.shape
        outarr = np.zeros((2*nz,2*nx,2*ny), dtype=arr.dtype)
        arrf = fftn(arr)
        arro = self.pad(arrf)
        outarr = ifftn(arro)
        return outarr

    def pad(self,arr):
        nz,nx,ny = arr.shape
        out = np.zeros((2*nz,2*nx,2*nx),arr.dtype)
        nxh = nx/2
        if (nz%2==0):
            nzh = nz/2
            out[:nzh,:nxh,:nxh] = arr[:nzh,:nxh,:nxh]
            out[:nzh,:nxh,3*nxh:4*nxh] = arr[:nzh,:nxh,nxh:nx]
            out[:nzh,3*nxh:4*nxh,:nxh] = arr[:nzh,nxh:nx,:nxh]
            out[:nzh,3*nxh:4*nxh,3*nxh:4*nxh] = arr[:nzh,nxh:nx,nxh:nx]
            out[3*nzh:4*nzh,:nxh,:nxh] = arr[nzh:nz,:nxh,:nxh]
            out[3*nzh:4*nzh,:nxh,3*nxh:4*nxh] = arr[nzh:nz,:nxh,nxh:nx]
            out[3*nzh:4*nzh,3*nxh:4*nxh,:nxh] = arr[nzh:nz,nxh:nx,:nxh]
            out[3*nzh:4*nzh,3*nxh:4*nxh,3*nxh:4*nxh] = arr[nzh:nz,nxh:nx,nxh:nx]
        else:
            nzh = nz/2
            out[:nzh,:nxh,:nxh] = arr[:nzh,:nxh,:nxh]
            out[:nzh,:nxh,3*nxh:4*nxh] = arr[:nzh,:nxh,nxh:nx]
            out[:nzh,3*nxh:4*nxh,:nxh] = arr[:nzh,nxh:nx,:nxh]
            out[:nzh,3*nxh:4*nxh,3*nxh:4*nxh] = arr[:nzh,nxh:nx,nxh:nx]
            out[(3*nzh+1):(2*nz),:nxh,:nxh] = arr[nzh:nz,:nxh,:nxh]
            out[(3*nzh+1):(2*nz),:nxh,3*nxh:4*nxh] = arr[nzh:nz,:nxh,nxh:nx]
            out[(3*nzh+1):(2*nz),3*nxh:4*nxh,:nxh] = arr[nzh:nz,nxh:nx,:nxh]
            out[(3*nzh+1):(2*nz),3*nxh:4*nxh,3*nxh:4*nxh] = arr[nzh:nz,nxh:nx,nxh:nx]
        return out