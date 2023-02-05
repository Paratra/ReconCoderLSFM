# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:49:21 2018

@author: rl74173
"""

import si_recon_2D_p36_n as si
import tifffile as tf

fns = r'C:/Users/linruizhe/Desktop/Substack (91-99).tif'
img = tf.imread(fns)

p = si.si2D(img,3,3,0.515,1.42)

p.mu = 0.012
p.fwhm = 0.99
p.strength = 0.00001
p.minv = 0.
p.eta = 0.06

p.separate(0)
p.shift0()
x1 = p.mapoverlap(0.001, 0.250, nps=8, r_ang=0.04, r_sp=0.04)
x1 = p.mapoverlap(x1[0], x1[1], nps=10, r_ang=0.005, r_sp=0.005)
x1 = p.mapoverlap(x1[0], x1[1], nps=10, r_ang=0.005, r_sp=0.005)

p.separate(1)
p.shift0()
x2 = p.mapoverlap(4.199, 0.248, nps=8, r_ang=0.04, r_sp=0.04)
x2 = p.mapoverlap(x2[0], x2[1], nps=10, r_ang=0.005, r_sp=0.005)
x2 = p.mapoverlap(x2[0], x2[1], nps=10, r_ang=0.005, r_sp=0.005)

p.separate(2)
p.shift0()
x3 = p.mapoverlap(2.091, 0.247, nps=8, r_ang=0.04, r_sp=0.04)
x3 = p.mapoverlap(x3[0], x3[1], nps=10, r_ang=0.005, r_sp=0.005)
x3 = p.mapoverlap(x3[0], x3[1], nps=10, r_ang=0.005, r_sp=0.005)
