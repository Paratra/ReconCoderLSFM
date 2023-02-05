# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:31:51 2020

2D Nonlinear SIM reconstruction (python3.7)

@author: spexophis
"""

from pylab import imshow, subplot, figure, colorbar
import psfsim36 as psfsim
import numpy as np
import tifffile as tf
from scipy import signal
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift

class si2D(object):
    
    def __init__(self,fns,nangles,nphs,wavelength,na):
        self.dx = 0.089
        self.na = na
        self.nph = nphs
        self.nang = nangles
        self.wl = wavelength
        self.norders = 5
        self.img = tf.imread(fns)
        nz, nx, ny = self.img.shape
        self.nx = nx
        self.ny = ny        
        self.img = self.img.reshape(self.nang,self.nph,nx,ny)
        self.psf = self.getpsf()
        self.meshgrid()
        self.sepmat = self.sepmatrix()
        self.mu = 0.02
        self.cutoff = 0.001
        self.strength = 1.
        self.sigma = 8.
        self.eh = []
        self.eta = 0.08
        self.winf = self.window(self.eta)
        
    def getpsf(self):
        dx = self.dx/4
        nx = self.nx*4
        psf = psfsim.psf()
        psf.setParams(wl=self.wl,na=self.na,dx=dx,nx=nx)
        self.radius = psf.radius
        psf.getFlatWF()
        psf1 = np.abs((fft2(fftshift(psf.bpp))))**2
        psf1 = psf1/psf1.sum()
        return psf1

    def meshgrid(self):
        nx = self.nx
        ny = self.ny
        x = np.arange(4*nx)
        y = np.arange(4*ny)
        xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
        i_xv = xv[1:2*nx+1,0]
        xv[2*nx:4*nx,0] = -np.flip(i_xv,0)
        i_yv = yv[0,1:2*ny+1]
        yv[0,2*ny:4*ny] = -np.flip(i_yv,0)
        self.xv = xv
        self.yv = yv
    
    def sepmatrix(self):
        nphases = self.nph
        norders = self.norders
        sepmat = np.zeros((norders,nphases),dtype=np.float32)
        norders = int((norders+1)/2)
        phi = 2*np.pi/nphases
        for j in range(nphases):
            sepmat[0, j] = 1.0/nphases
            for order in range(1,norders):
                sepmat[2*order-1,j] = 2.0 * np.cos(j*order*phi)/nphases
                sepmat[2*order  ,j] = 2.0 * np.sin(j*order*phi)/nphases
        return sepmat

    def separate(self,nangle):
        Nw = self.nx
        outr = np.dot(self.sepmat,self.img[nangle].reshape(self.nph,Nw**2))
        self.separr = np.zeros((self.norders,Nw*4,Nw*4),dtype=np.complex64)
        self.separr[0]=np.fft.fftshift(self.interp(outr[0].reshape(Nw,Nw))*self.winf)
        self.separr[1]=np.fft.fftshift(self.interp((outr[1]+1j*outr[2]).reshape(Nw,Nw))*self.winf)
        self.separr[2]=np.fft.fftshift(self.interp((outr[1]-1j*outr[2]).reshape(Nw,Nw))*self.winf)
        self.separr[3]=np.fft.fftshift(self.interp((outr[3]+1j*outr[4]).reshape(Nw,Nw))*self.winf)
        self.separr[4]=np.fft.fftshift(self.interp((outr[3]-1j*outr[4]).reshape(Nw,Nw))*self.winf)
        return True

    def shift0(self):
        self.otf0 = fft2(self.psf)
        zsp = self.zerosuppression(0,0)
        self.otf0 = self.otf0 * zsp
        self.imgf0 = fft2(self.separr[0])
        return True
        
    def shift1(self,angle,spacing):
        ''' shift data in freq space by multiplication in real space '''
        Nw = self.nx * 2
        dx = self.dx / 4
        kx = dx*np.cos(angle)/spacing
        ky = dx*np.sin(angle)/spacing
        #shift matrix
        ysh = np.zeros((2,2*Nw,2*Nw), dtype=np.complex64)
        ysh[0,:,:] = np.exp(2j*np.pi*(kx*self.xv+ky*self.yv))
        ysh[1,:,:] = np.exp(2j*np.pi*(-kx*self.xv-ky*self.yv))
        #1st order positive
        self.otf_1_0 = fft2(self.psf*ysh[0])
        yshf = np.abs(fft2(ysh[0]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<Nw):
            sx = sx
        else:
            sx = sx-2*Nw
        if (sy<Nw):
            sy = sy
        else:
            sy = sy-2*Nw
        temp = np.sqrt(sx**2+sy**2) / (2 * self.radius)
        self.eh = np.append(self.eh,temp)
        zsp = self.zerosuppression(sx,sy)
        self.otf_1_0 = self.otf_1_0 * zsp
        self.imgf_1_0 = fft2(self.separr[1]*ysh[0])
        #1st order negative
        self.otf_1_1 = fft2(self.psf*ysh[1])
        yshf = np.abs(fft2(ysh[1]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<Nw):
            sx = sx
        else:
            sx = sx-2*Nw
        if (sy<Nw):
            sy = sy
        else:
            sy = sy-2*Nw
        zsp = self.zerosuppression(sx,sy)
        temp = np.sqrt(sx**2+sy**2) / (2 * self.radius)
        self.eh = np.append(self.eh,temp)
        self.otf_1_1 = self.otf_1_1 * zsp
        self.imgf_1_1 = fft2(self.separr[2]*ysh[1])
        return True
    
    def shift2(self,angle,spacing):
        ''' shift data in freq space by multiplication in real space '''
        Nw = self.nx * 2
        dx = self.dx / 4
        kx = dx*np.cos(angle)/spacing
        ky = dx*np.sin(angle)/spacing
        #shift matrix
        ysh = np.zeros((2,2*Nw,2*Nw), dtype=np.complex64)
        ysh[0,:,:] = np.exp(2j*np.pi*(kx*self.xv+ky*self.yv))
        ysh[1,:,:] = np.exp(2j*np.pi*(-kx*self.xv-ky*self.yv))
        #1st order positive
        self.otf_2_0 = fft2(self.psf*ysh[0])
        yshf = np.abs(fft2(ysh[0]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<Nw):
            sx = sx
        else:
            sx = sx-2*Nw
        if (sy<Nw):
            sy = sy
        else:
            sy = sy-2*Nw
        temp = np.sqrt(sx**2+sy**2) / (2 * self.radius)
        self.eh = np.append(self.eh,temp)
        zsp = self.zerosuppression(sx,sy)
        self.otf_2_0 = self.otf_2_0 * zsp
        self.imgf_2_0 = fft2(self.separr[3]*ysh[0])
        #1st order negative
        self.otf_2_1 = fft2(self.psf*ysh[1])
        yshf = np.abs(fft2(ysh[1]))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<Nw):
            sx = sx
        else:
            sx = sx-2*Nw
        if (sy<Nw):
            sy = sy
        else:
            sy = sy-2*Nw
        zsp = self.zerosuppression(sx,sy)
        temp = np.sqrt(sx**2+sy**2) / (2 * self.radius)
        self.eh = np.append(self.eh,temp)
        self.otf_2_1 = self.otf_2_1 * zsp
        self.imgf_2_1 = fft2(self.separr[4]*ysh[1])
        return True

    def getoverlap(self,angle,spacing,verbose=False):
        ''' shift 2nd order data '''
        dx = self.dx / 4
        Nw = self.nx * 2
        kx = dx*np.cos(angle)/spacing
        ky = dx*np.sin(angle)/spacing

        ysh = np.exp(2j*np.pi*(kx*self.xv+ky*self.yv))
        otf = fft2(self.psf*ysh)
        yshf = np.abs(fft2(ysh))
        sx, sy = np.unravel_index(yshf.argmax(), yshf.shape)
        if (sx<Nw):
            sx = sx
        else:
            sx = sx-2*Nw
        if (sy<Nw):
            sy = sy
        else:
            sy = sy-2*Nw
        zsp = self.zerosuppression(sx,sy)
        otf = otf * zsp
        imgf = fft2(self.separr[1]*ysh)
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf*imgf0
        wimgf1 = otf0*imgf
        msk = abs(otf0*otf)>cutoff
        a = np.sum(msk*wimgf1*wimgf0.conj())/np.sum(msk*wimgf0*wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
        if verbose:
            t = (msk*wimgf1*wimgf0.conj())/(msk*wimgf0*wimgf0.conj()) 
            t[np.isnan(t)] = 0.0
            figure()
            imshow(np.abs(np.fft.fftshift(t)), interpolation='nearest', vmin=0.0, vmax=2.0)
            colorbar()
            figure()
            imshow(np.angle(np.fft.fftshift(t)), interpolation='nearest')
            colorbar()
        return mag, phase

    def mapoverlap(self,angle,spacing,nps=10,r_ang=0.02,r_sp=0.008):
        d_ang = 2*r_ang/nps
        d_sp = 2*r_sp/nps
        ang_iter = np.arange(-r_ang,r_ang+d_ang/2,d_ang)+angle
        sp_iter = np.arange(-r_sp,r_sp+d_sp/2,d_sp)+spacing
        magarr = np.zeros((nps+1,nps+1))
        pharr = np.zeros((nps+1,nps+1))
        for m,ang in enumerate(ang_iter):
            for n,sp in enumerate(sp_iter):
                print (m,n)
                mag, phase = self.getoverlap(ang,sp)
                if np.isnan(mag):
                    magarr[m,n] = 0.0
                else:
                    magarr[m,n] = mag
                    pharr[m,n] = phase
        figure()
        subplot(211)
        imshow(magarr,interpolation='nearest')
        subplot(212)
        imshow(pharr,interpolation='nearest')
        # get maximum
        k, l = np.where( magarr == magarr.max())
        angmax = k[0]*d_ang - r_ang + angle
        spmax = l[0]*d_sp - r_sp + spacing
        return (angmax,spmax,magarr.max())
    
    def recon1(self,n,ang,spacing,phase,mag):
        # construct 1 angle
        nx = 4*self.nx
        ny = 4*self.ny
        mu = self.mu
        
        imgf = np.zeros((nx,ny),dtype=np.complex64)
        otf = np.zeros((nx,ny),dtype=np.complex64)        
        
        self.Snum = np.zeros((nx,ny),dtype=np.complex64)
        self.Sden = np.zeros((nx,ny),dtype=np.complex64)
        self.Sden += mu**2
        
        for i in range(n):
            self.separate(i)
            self.shift0()
            self.shift1(ang[i],spacing[i])
            self.shift2(ang[i],spacing[i]/2.)
            ph1 = mag[i]*np.exp(1j*phase[i])
            ph2 = mag[i]*np.exp(1j*2.*phase[i])
            # 0th order
            imgf = self.imgf0
            otf = self.otf0
            self.Snum += otf.conj()*imgf
            self.Sden += abs(otf)**2
            # +1st order
            imgf = self.imgf_1_0
            otf = self.otf_1_0
            self.Snum += ph1*otf.conj()*imgf
            self.Sden += abs(otf)**2
            # -1st order
            imgf = self.imgf_1_1
            otf = self.otf_1_1
            self.Snum += ph1.conj()*otf.conj()*imgf
            self.Sden += abs(otf)**2
            # +2nd order
            imgf = self.imgf_2_0
            otf = self.otf_2_0
            self.Snum += ph2*otf.conj()*imgf
            self.Sden += abs(otf)**2
            # -2nd order
            imgf = self.imgf_2_1
            otf = self.otf_2_1
            self.Snum += ph2.conj()*otf.conj()*imgf
            self.Sden += abs(otf)**2
            # # finish
            # self.S = self.Snum/self.Sden
            # self.finalimage = np.fft.fftshift(ifft2(S))
        ss = 2. * np.max(self.eh)
        A = self.apod(ss)
        self.S = self.Snum/self.Sden * A
        self.finalimage = fftshift(ifft2(self.S))
        tf.imsave('final_image.tif', self.finalimage.real.astype(np.float32),photometric='minisblack')
        tf.imsave('effective_OTF.tif',np.abs(fftshift(self.S)).astype(np.float32),photometric='minisblack')
        return True
    
    def interp(self,arr):
        nx,ny = arr.shape
        px = int(3*nx/2)
        py = int(3*ny/2)
        arrf = fft2(arr)
        arro = np.pad(np.fft.fftshift(arrf),((px,px),(py,py)),'constant', constant_values=(0))
        out = ifft2(np.fft.fftshift(arro))
        return out

    def window(self,eta):
        nx = self.nx * 4
        wxy = signal.tukey(nx, alpha=eta, sym=True)
        wx = np.tile(wxy,(nx,1))
        wy = wx.swapaxes(0,1)
        w = wx * wy
        return w
    
    def zerosuppression(self,sx,sy):
        x = self.xv
        y = self.yv
        g = 1 - self.strength * np.exp(-((x-sx)**2.+(y-sy)**2.)/(2.*self.sigma**2.))
        g[g<0.5] = 0.0
        g[g>=0.5] = 1.0
        return g
    
    def apod(self,mp):
        r = (2*mp)*self.radius
        ap = 1 - np.sqrt(self.xv**2 + self.yv**2) / r
        ap[ap<0.] = 0.
        return ap
