# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:15:59 2013

@author: kner
"""

import Utility36 as U
import Zernike36 as Z

import tifffile as tf
import numpy as N
import numpy.random as rd
pi = N.pi
fft2 = N.fft.fft2
ifft2 = N.fft.ifft2
fftn = N.fft.fftn
ifftn = N.fft.ifftn
fftshift = N.fft.fftshift

class sim(object):
    
    def __init__(self):
        self.Np = 512 # no. of particles
        self.dx = 0.04
        self.nx = 512
        self.na = 1.42
        self.wl = 0.51
        self.sp = 0.75#self.wl/self.na/2.
        self.nzarr = 15
        self.zarr = N.zeros(15)
        self.img = N.zeros((self.nx,self.nx))
    
    def __del__(self):
        pass
    
    def getaberr(self):
        wl = self.wl
        na = self.na
        n2 = 1.512
        dp = 1/(self.nx*self.dx)
        radius = (na/wl)/dp
        msk = U.shift(U.discArray((self.nx,self.nx),radius))/N.sqrt(pi*radius**2)/self.nx
        phi = N.zeros((self.nx,self.nx))
        for m in range(1,self.nzarr):
            phi = phi + self.zarr[m]*Z.Zm(m,radius,[0,0],self.nx)
        self.wf = msk*N.exp(1j*phi).astype(N.complex64)

    def getobj(self):
        ''' put fluorophores randomly or in circle '''
        dx = self.dx
        nxh = int(self.nx/2)
        Np = self.Np
#        Np = 2
#        self.xps[0] = nxh*dx
#        self.yps[0] = nxh*dx
#        self.xps[1] = self.xps[0] + 0.1    
#        self.yps[1] = self.ypx[0]
#        rad = 0.25
#        phi = N.linspace(0,2*pi,Np)
#        self.xps = rad*N.cos(phi) + nxh*dx
#        self.yps = rad*N.sin(phi) + nxh*dx
#         self.xps = (self.dx*self.nx)*(0.8*rd.rand(Np)+0.1)
#         self.yps = (self.dx*self.nx)*(0.8*rd.rand(Np)+0.1)
        xx = N.linspace(0.2, 0.8, self.nx)
        xps1 = (self.dx * self.nx) * (xx)
        yps1 = (self.dx * self.nx) * (0.9 * xx)
        xps2 = (self.dx * self.nx) * (xx+0.1)
        yps2 = (self.dx * self.nx) * (xx+0.1)
        xps3 = (self.dx * self.nx) * (xx)
        yps3 = (self.dx * self.nx) * (0.1 * (xx - 0.5) * (xx - 0.5))
        xps4 = (self.dx * self.nx) * (xx)
        yps4 = (self.dx * self.nx) * (0.2 * (xx - 0.5) * (xx - 0.5))
        argsx = (xps1, xps2, xps3, xps4)
        argsy = (yps1, yps2, yps3, yps4)
        self.xps = N.concatenate(argsx)
        self.yps = N.concatenate(argsy)
        
    def getLineobj(self):
        ''' draw a series of random lines '''
        xps = []
        yps = []
        Nl = 12 # # of lines
        x1 = (self.dx*self.nx)*(0.8*rd.rand(Nl)+0.1)
        y1 = (self.dx*self.nx)*(0.8*rd.rand(Nl)+0.1)
        x2 = (self.dx*self.nx)*(0.8*rd.rand(Nl)+0.1)
        y2 = (self.dx*self.nx)*(0.8*rd.rand(Nl)+0.1)
        for m in range(Nl):
            dl = N.sqrt((x2[m]-x1[m])**2+(y2[m]-y1[m])**2)
            Np = int(dl/0.02)
            s = N.linspace(0,1,Np)
            xls = [(x2[m]-x1[m])*t+x1[m] for t in s]
            yls = [(y2[m]-y1[m])*t+y1[m] for t in s]
            xps.append(xls)
            yps.append(yls)
        self.xps = N.array([x for x in xls for xls in xps])
        self.yps = N.array([y for y in yls for yls in yps])
        self.Np = self.xps.shape[0]
        return True

    def addpsf(self,x,y,I):
        # create phase
        nx = self.nx
        alpha = 2*pi/nx/self.dx
        g = lambda m, n: N.exp(1j*alpha*(m*x+n*y)).astype(N.complex64)
        ph = N.fromfunction(g, (nx,nx), dtype=N.float32)
        ph = U.shift(ph)
        wfp = N.sqrt(I)*ph*self.wf
        #wfp = wfp
        self.img = self.img + abs(fft2(wfp))**2

    def getoneimg(self,angle,phase,Iph):
        self.img[:, :] = 20.0
        # get points        
        # create psfs
        kx = 2 * pi * N.cos(angle) / self.sp
        ky = 2 * pi * N.sin(angle) / self.sp
        for m in range(4*self.nx):
            Ip = Iph * 0.5 * (1 + N.cos(kx * self.xps[m] + ky * self.yps[m] + phase))
            self.addpsf(self.xps[m], self.yps[m], Ip)
        # noise
        self.img = rd.poisson(self.img)
        # done!

    def getoneimg_aberration(self,angle,phase,Iph):
        self.img[:,:] = 20.0
        # get points
        # create psfs
        kx = 2*pi*N.cos(angle)/self.sp
        ky = 2*pi*N.sin(angle)/self.sp
        for m in range(self.Np):
            Ip = Iph*0.5*(1+N.cos(kx*self.xps[m]+ky*self.yps[m]+phase))
            self.addpsf(self.xps[m],self.yps[m],Ip)
        # noise
        self.img = rd.poisson(self.img)
        # done!
        
    def runoneangle(self,Iph=1000,aberration=False):
        out = N.zeros((3,self.nx,self.nx),dtype=N.float32)
        if aberration:
            self.getoneimg_aberration(0.0,0.0,Iph)
            out[0,:,:] = self.img
            self.getoneimg_aberration(0.0,2*pi/3,Iph)
            out[1,:,:] = self.img
            self.getoneimg_aberration(0.0,4*pi/3,Iph)
            out[2,:,:] = self.img
        else:
            self.getoneimg(0.0, 0.0, Iph)
            out[0, :, :] = self.img
            self.getoneimg(0.0, 2 * pi / 3, Iph)
            out[1, :, :] = self.img
            self.getoneimg(0.0, 4 * pi / 3, Iph)
            out[2, :, :] = self.img
        tf.imsave('sim_si2d_oneagnle.tif',out,photometric='minisblack')
        
    def runthreeangles(self,Iph=1000,fn='sim_si2d_threeangles.tif'):
        out = N.zeros((9,self.nx,self.nx),dtype=N.float32)
        for m in range(3):
            theta = m*(2*pi/3)
            self.getoneimg(theta,0.0,Iph)
            out[3*m,:,:] = self.img
            self.getoneimg(theta,2*pi/3,Iph)
            out[3*m+1,:,:] = self.img
            self.getoneimg(theta,4*pi/3,Iph)
            out[3*m+2,:,:] = self.img
        tf.imsave(fn,out,photometric='minisblack')

s = sim()
s.getaberr()
s.getobj()
s.runoneangle(Iph=1000,aberration=True)
