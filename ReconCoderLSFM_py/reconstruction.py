# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 10:42:22 2018
@author: rl74173
"""
import si_recon_2D_p36_n as si
import tifffile as tf
import numpy as np
#data file
fns = r'C:/Users/linruizhe/Desktop/Substack (91-99).tif'
img = tf.imread(fns)
#1st angle parameters
x1 = np.array([0.003, 0.25])
#2nd angle parameters
x2 = np.array([4.2, 0.248])
#3rd angle parameters
x3 = np.array([2.091, 0.247])
#reconstruction
p = si.si2D(img, 3, 3, 0.515, 1.42)
#wiener filter
p.mu = 0.1
p.fwhm = 0.99
p.strength = 0.00001
p.minv = 0.
p.eta = 0.06
#1st angle
p.separate(0)
p.shift0()
p.shift(x1[0], x1[1])
a1 = p.getoverlap(x1[0], x1[1])
#2nd angle
p.separate(1)
p.shift0()
p.shift(x2[0], x2[1])
a2 = p.getoverlap(x2[0], x2[1])
#3rd angle
p.separate(2)
p.shift0()
p.shift(x3[0], x3[1])
a3 = p.getoverlap(x3[0], x3[1])
#ang,spacing,phase,mag
p.recon1(3,[x1[0],x2[0],x3[0]],[x1[1],x2[1],x3[1]],[-a1[1],-a2[1],-a3[1]],[1.,1.,1.])#[a1[0],a2[0],a3[0]]
#save image and otf
tf.imsave('final_image.tif',np.abs(p.finalimage).astype(np.float32),photometric='minisblack')
tf.imsave('effective_OTF.tif',np.abs(np.fft.fftshift(p.Snum/p.Sden)).astype(np.float32),photometric='minisblack')