

from pylab import imshow, subplot, figure, colorbar
import psfsim36 as psfsim
import numpy as np
import tifffile as tf
from scipy import signal

from pdb import set_trace as st

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
        self.img = self.subback(img_stack)
        self.img = self.img.reshape(self.nang,self.nph,nx,ny)
        self.psf = self.getpsf()
        self.meshgrid()
        
        self.strength = 0.99
        self.fwhm = 0.00001
        self.minv = 0.0
        self.eh = []
        self.eta = 0.08
        self.winf = self.window(self.eta)
        
    def subback(self,img):
        nz, nx, ny = img.shape
        for i in range(nz):
            data = img[i,:,:]
            hist, bin_edges = np.histogram(img[i,:,:], bins=np.arange(img[i,:,:].min(),img[i,:,:].max()) )
            
            ind = np.where(hist == hist.max())
            bg = bin_edges[np.max(ind)+1]

            

            data[data<=bg] = 0.
            data[data>bg] = data[data>bg] - bg
            img[i,:,:] = data

        return img
        
    def getpsf(self):
        dx = self.dx/2
        nx = self.nx*2
        psf = psfsim.psf()
        psf.setParams(wl=self.wl,na=self.na,dx=dx,nx=nx)
        self.radius = psf.radius
        psf.getFlatWF()
        wf = psf.bpp
        psf1 = np.abs((fft2(wf)))**2
        psf1 = psf1/psf1.sum()
        st()
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
        outr = np.dot(mat,self.img[nangle].reshape(self.nph,Nw**2))
        self.separr = np.zeros((3,Nw*2,Nw*2),dtype=np.complex64)
        self.separr[0]=np.fft.fftshift(self.interp(outr[0].reshape(Nw,Nw))*self.winf)
        self.separr[1]=np.fft.fftshift(self.interp((outr[1]+1j*outr[2]).reshape(Nw,Nw))*self.winf)
        self.separr[2]=np.fft.fftshift(self.interp((outr[1]-1j*outr[2]).reshape(Nw,Nw))*self.winf)
        return True

    def shift0(self):
        self.otf0 = fft2(self.psf)
        zsp = self.zerosuppression(0,0)
        self.otf0 = self.otf0 * zsp
        self.imgf0 = fft2(self.separr[0])
        return True
        
    def shift(self,angle,spacing):
        ''' shift data in freq space by multiplication in real space '''
        Nw = self.nx
        dx = self.dx / 2
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

    def getoverlap(self,angle,spacing,verbose=False):
        ''' shift 2nd order data '''
        dx = self.dx / 2
        Nw = self.nx
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
            self.S = self.Snum/self.Sden
            #self.finalimage = np.fft.fftshift(ifft2(S))
#        ss = np.max(self.eh)
#        A = self.apod(ss)
        self.S = self.Snum/self.Sden #* A
        self.finalimage = np.fft.fftshift(ifft2(self.S))
        tf.imsave('final_image.tif',np.abs(self.finalimage).astype(np.float32),photometric='minisblack')
        tf.imsave('effective_OTF.tif',np.abs(np.fft.fftshift(self.S)).astype(np.float32),photometric='minisblack')
        return True
    
    def interp(self,arr):
        nx,ny = arr.shape
        px = int(nx/2)
        py = int(ny/2)
        arrf = fft2(arr)
        arro = np.pad(np.fft.fftshift(arrf),((px,px),(py,py)),'constant', constant_values=(0))
        out = ifft2(np.fft.fftshift(arro))
        return out

    def window(self,eta):
        nx = self.nx * 2
        #ny = self.ny * 2
        wxy = signal.tukey(nx, alpha=eta, sym=True)
        wx = np.tile(wxy,(nx,1))
        wy = wx.swapaxes(0,1)
        w = wx * wy
        return w
    
    def zerosuppression(self,sx,sy):
        x = self.xv
        y = self.yv
        g = 1 - self.strength*np.exp(-(np.abs((x-sx)**2)+np.abs((y-sy)**2)))/(2*(self.fwhm**2))
        g[g<1.0] = self.minv
        return g
    
    def apod(self,mp):
        r = (2*mp)*self.radius
        ap = 1 - np.sqrt(self.xv**2 + self.yv**2) / r
        ap[ap<0.] = 0.
        return ap
    