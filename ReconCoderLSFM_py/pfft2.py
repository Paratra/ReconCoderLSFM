# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:28:31 2017

@author: rl74173
"""

from pylab import * 
#This function returns the reciprocal space P and S components 
def mfft2(im):
  [rows,cols] = shape(im) 
  #Compute boundary conditions 
  s = zeros( shape(im) ) 
  s[0,0:] = im[0,0:] - im[rows-1,0:] 
  s[rows-1,0:] = -s[0,0:] 
  s[0:,0] = s[0:,0] + im[0:,0] - im[:,cols-1] 
  s[0:,cols-1] = s[0:,cols-1] - im[0:,0] + im[:,cols-1] 
  #Create grid for computing Poisson solution 
  [cx, cy] = meshgrid(2*pi*arange(0,cols)/cols, 2*pi*arange(0,rows)/rows) 
  #Generate smooth component from Poisson Eq with boundary condition 
  D = (2*(2 - cos(cx) - cos(cy))) 
  D[0,0] = inf # Enforce 0 mean & handle div by zero 
  S = fft2(s)/D 
  P = fft2(im) - S # FFT of periodic component 
  return P