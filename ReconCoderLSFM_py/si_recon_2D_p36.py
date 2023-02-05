''' 2D structured illumination reconstruction
    need to estimate mu from data 
    
    python3.6 Ruizhe Lin 06/08/2018 '''

import os, sys, time
sys.path.append('..\\simsxy')
sys.path.append('..\\storm')
sys.path.append('..\\guitest')

from pylab import imshow, subplot, figure
import psfsim36 as psfsim
import numpy as np
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2

class si2D(object):
    
    def __init__(self,img_stack,nangles,nphs,wavelength,na):
        self.dx = 0.135
        self.na = na
        self.nph = nphs
        self.nang = nangles
        nz, nx, ny = img_stack.shape
        self.nx = nx
        self.ny = ny
        self.mu = 0.01
        self.cutoff = 0.001
        self.wl = wavelength
        self.img = img_stack.reshape(self.nang,self.nph,nx,ny)
        self.psf = self.getpsf()
        self.meshgrid()
   
    def getpsf(self):
        dx = self.dx/2
        nx = self.nx*2
        psf = psfsim.psf()
        psf.setParams(wl=self.wl,na=self.na,dx=dx,nx=nx)      
        psf.getFlatWF()
        wf = psf.bpp
        psf1 = np.abs((fft2(wf)))**2
        psf1 = psf1/psf1.sum()
        return psf1

    def meshgrid(self):
        nx = self.nx
        ny = self.ny
        x = np.arange(2*nx)
        y = np.arange(2*ny)
        xv, yv = np.meshgrid(x, y, indexing='ij', sparse=True)
        i_xv = xv[1:nx+1,0]
        xv[nx:2*nx,0] = -np.flip(i_xv,0)
        i_yv = yv[0,1:ny+1]
        yv[0,ny:2*ny] = -np.flip(i_yv,0)
        self.xv = xv
        self.yv = yv

    def shiftmat(self,kx,ky):
        xv = self.xv
        yv = self.yv
        a = np.exp(2j*np.pi*(kx*xv+ky*yv))
        return a

    def sepmatrix(self):
        nphases = self.nph
        sepmat = np.zeros((3, nphases), dtype=np.float32)
        norders = 2
        phi = 2*np.pi/nphases;
        for j in range(nphases):
            sepmat[0, j] = 1.0/nphases
            for order in range(1,norders):
                sepmat[2*order-1,j] = 2.0 * np.cos(j*order*phi)/nphases
                sepmat[2*order  ,j] = 2.0 * np.sin(j*order*phi)/nphases
        return sepmat

    def separate(self,nangle):
        Nw = self.nx
        mat = self.sepmatrix()
        outr = np.dot(mat,self.img[::nangle].reshape(self.nph,Nw**2))
        self.separr = np.zeros((3,Nw*2,Nw*2),dtype=np.complex64)
        self.separr[0]=np.fft.fftshift(self.interp(outr[0].reshape(Nw,Nw)))
        self.separr[1]=np.fft.fftshift(self.interp((outr[1]+1j*outr[2]).reshape(Nw,Nw)))
        self.separr[2]=np.fft.fftshift(self.interp((outr[1]-1j*outr[2]).reshape(Nw,Nw)))
        return True

    def shift0(self):
        Nw = self.nx
        #0 order
        self.otf0 = np.zeros((2*Nw,2*Nw),dtype=np.complex64)
        self.otf0[:] = fft2(self.psf)
        self.imgf0 = np.zeros((2*Nw,2*Nw),dtype=np.complex64)
        self.imgf0[:] = fft2(self.separr[0])
        return True
        
    def shift(self,angle,spacing):
        ''' shift data in freq space by multiplication in real space '''
        Nw = self.nx
        dx = self.dx / 2
        kx = dx*np.cos(angle)/spacing
        ky = dx*np.sin(angle)/spacing
        #shift matrix
        ysh = np.zeros((2,2*Nw,2*Nw), dtype=np.complex64)
        ysh[0,:,:] = self.shiftmat(kx,ky)
        ysh[1,:,:] = self.shiftmat(-kx,-ky)
        #1st order positive
        self.otf_1_0 = np.zeros((2*Nw,2*Nw),dtype=np.complex64)
        self.otf_1_0[:] = fft2(self.psf*ysh[0])
        self.imgf_1_0 = np.zeros((2*Nw,2*Nw),dtype=np.complex64)
        self.imgf_1_0[:] = fft2(self.separr[1]*ysh[0])
        #1st order negative
        self.otf_1_1 = np.zeros((2*Nw,2*Nw),dtype=np.complex64)
        self.otf_1_1[:] = fft2(self.psf*ysh[1])
        self.imgf_1_1 = np.zeros((2*Nw,2*Nw),dtype=np.complex64)
        self.imgf_1_1[:] = fft2(self.separr[2]*ysh[1])
        return True

    def getoverlap(self,angle,spacing):
        ''' shift 2nd order data '''
        dx = self.dx / 2
        kx = dx*np.cos(angle)/spacing
        ky = dx*np.sin(angle)/spacing
        
        ysh = self.shiftmat(kx,ky)
        otf = fft2(self.psf*ysh)
        
        imgf = self.separr[1].astype(np.complex64)
        imgf = fft2(imgf*ysh)
        
        cutoff = self.cutoff
        imgf0 = self.imgf0
        otf0 = self.otf0
        wimgf0 = otf*imgf0
        wimgf1 = otf0*imgf
        msk = (abs(otf0*otf)>cutoff).astype(np.complex64)
        a = np.sum(msk*wimgf1*wimgf0.conj())/np.sum(msk*wimgf0*wimgf0.conj())
        mag = np.abs(a)
        phase = np.angle(a)
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
        nx = 2*self.nx
        ny = 2*self.ny
        mu = self.mu
        
        imgf = np.zeros((nx,ny),dtype=np.complex64)
        otf = np.zeros((nx,ny),dtype=np.complex64)        
        
        self.Snum = np.zeros((nx,ny),dtype=np.complex64)
        self.Sden = np.zeros((nx,ny),dtype=np.complex64)
        self.Sden += mu**2
        
        for i in range(n):
            self.separate(i)
            self.shift0()
            self.shift(ang[i],spacing[i])
            ph = mag[i]*np.exp(1j*phase[i])
            # 0th order
            imgf = self.imgf0
            otf = self.otf0
            self.Snum += otf.conj()*imgf
            self.Sden += abs(otf)**2
            # +1st order
            imgf = self.imgf_1_0
            otf = self.otf_1_0
            self.Snum += ph*otf.conj()*imgf
            self.Sden += abs(otf)**2
            # -1 order
            imgf = self.imgf_1_1
            otf = self.otf_1_1
            self.Snum += ph.conj()*otf.conj()*imgf
            self.Sden += abs(otf)**2
            # finish
            S = self.Snum/self.Sden
            self.finalimage = np.fft.fftshift(ifft2(S))
        return True
    
    def pad(self, arr):
        nx,ny = arr.shape
        out = np.zeros((2*nx,2*ny),arr.dtype)
        nxh = np.int(nx/2)
        out[:nxh,:nxh] = arr[:nxh,:nxh]
        out[:nxh,3*nxh:4*nxh] = arr[:nxh,nxh:nx]
        out[3*nxh:4*nxh,:nxh] = arr[nxh:nx,:nxh]
        out[3*nxh:4*nxh,3*nxh:4*nxh] = arr[nxh:nx,nxh:nx]
        return out

    def interp(self, arr):
        nx,ny = arr.shape
        outarr = np.zeros((2*nx,2*ny),dtype=arr.dtype)
        arrf = fft2(arr)
        outarr = ifft2(self.pad(arrf))
        return outarr