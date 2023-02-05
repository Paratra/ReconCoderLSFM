# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:33:56 2019

@author: rl74173
"""

import numpy as np
from scipy import signal
import Utility36 as U
import tifffile as tf
import matplotlib.pyplot as plt

na = 1.2
wl = 0.67

def snr(wl,img):
    img = np.array(img)
    nx = 256
    img = img[128:384,128:384]
#    w = window(img)
#    img = img*w
    dp = 1/(nx*.089)
    radius = (na/wl)/dp
    hp = U.discArray((nx,nx),1.2*radius)-U.discArray((nx,nx),0.2*radius)
    aft = np.abs(np.fft.fftshift(np.fft.fft2(img))) * hp
    hpC = aft.sum()
    return aft, hpC

def window(img):
    nx,ny = img.shape
    wx = signal.tukey(nx, alpha=0.1)
    winx = np.tile(wx,(ny,1))
    winy = winx.swapaxes(0,1)
    win = winx * winy
    return win

fns=['C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm10_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm10_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm10_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm11_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm11_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm11_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm11_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm11_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm12_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm12_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm12_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm12_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm12_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm13_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm13_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm13_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm13_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm13_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm14_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm14_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm14_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm14_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm14_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm15_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm15_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm15_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm15_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm15_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm16_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm16_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm16_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm16_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm16_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm04_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm04_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm04_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm04_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm04_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm05_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm05_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm05_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm05_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm05_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm06_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm06_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm06_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm06_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm06_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm07_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm07_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm07_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm07_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm07_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm08_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm08_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm08_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm08_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm08_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm09_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm09_amp0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm09_amp-0.2000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm09_amp0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm09_amp-0.4000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm10_amp0.0000.tif', 'C:/Users/rl74173/Desktop/20191126_rl74173/20191126-132037_zm10_amp0.2000.tif']
fns.sort()

modes = np.arange(4,17)
amp = np.arange(-0.24,0.36,0.12)
mv = np.zeros(( modes.shape[0], amp.shape[0] ))
zv = np.zeros((modes.shape[0]))
z = np.zeros((5,512,512))
zf = np.zeros((5,256,256))

xp = np.linspace(-0.24, 0.24, 101)

for m, mode in enumerate(modes):
    z[0,:,:] = tf.imread(fns[5*m + 1])
    z[1,:,:] = tf.imread(fns[5*m + 0])
    z[2,:,:] = tf.imread(fns[5*m + 2])
    z[3,:,:] = tf.imread(fns[5*m + 3])
    z[4,:,:] = tf.imread(fns[5*m + 4])
    tf.imsave('z%.2d_img.tif'%(mode),z.astype(np.float32),photometric='minisblack')
    zf[0,:,:], mv[m,0] = snr(wl, z[0,:,:])
    zf[1,:,:], mv[m,1] = snr(wl, z[1,:,:])
    zf[2,:,:], mv[m,2] = snr(wl, z[2,:,:])
    zf[3,:,:], mv[m,3] = snr(wl, z[3,:,:])
    zf[4,:,:], mv[m,4] = snr(wl, z[4,:,:])
    tf.imsave('z%.2d_fft.tif'%(mode),zf.astype(np.float32),photometric='minisblack')

for i in range(modes.shape[0]):
#    p = np.poly1d(np.polyfit(amp, mv[i,:], 2))
#    plt.figure()
#    _ = plt.plot(amp, mv[i,:], '.', xp, p(xp), '-')
    a,b,c = np.polyfit(amp, mv[i,:], 2)
    zmax = -1*b/a/2.0
    zv[i] = zmax
plt.plot(modes,zv)
plt.figure()
plt.imshow(mv)

zzvv = np.zeros((modes.shape[0]))
for i in range(modes.shape[0]):
#    p = np.poly1d(np.polyfit(amp, mv[i,:], 2))
#    plt.figure()
#    _ = plt.plot(amp, mv[i,:], '.', xp, p(xp), '-')
    aa,bb,cc = np.polyfit(amp[::2], mv[i,::2], 2)
    zzmax = -1*bb/aa/2.0
    zzvv[i] = zzmax
plt.figure()
plt.plot(modes,zzvv)
plt.figure()
plt.imshow(mv[:,::2])

zzvv1 = np.zeros((modes.shape[0]))
for i in range(modes.shape[0]):
#    p = np.poly1d(np.polyfit(amp, mv[i,:], 2))
#    plt.figure()
#    _ = plt.plot(amp, mv[i,:], '.', xp, p(xp), '-')
    aa1,bb1,cc1 = np.polyfit(amp[1:4], mv[i,1:4], 2)
    zzmax1 = -1*bb1/aa1/2.0
    zzvv1[i] = zzmax1
plt.figure()
plt.plot(modes,zzvv1)
plt.figure()
plt.imshow(mv[:,1:4])
