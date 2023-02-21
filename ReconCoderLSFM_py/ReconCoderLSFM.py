# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 22:09:06 2022

@author: yangliu
"""
import si_recon_2D_p36_n as si
import tifffile as tf
import numpy as np
from scipy.ndimage import shift
from pdb import set_trace as st
import os
import time

start_time = time.time()


numangle = 3
numphase = 3
wl = 0.525
na = 0.8
size = 512
pixel_size = 0.135
angle1 = (-179.432+90) /360*2*np.pi #0.459
angle2 = (-116.338+90) /360*2*np.pi # -2.80
angle3 = (116.275+90) /360*2*np.pi #2.871
patternspacing1 = 1/(97.571*(1/(size*pixel_size))) #1.54
patternspacing2 = 1/(96.644*(1/(size*pixel_size)))
patternspacing3 = 1/(97.26*(1/(size*pixel_size)))

# st()     

from scipy import signal

def getWField(img,numphase):
    n = 0
    phase =np.arange(numphase)
    while len(img)>3:
        print(len(img))
        temp = np.mean(img[phase,:,:],0)
        # st()
        temp = temp[np.newaxis,:,:]
        if n ==0:
            wf = temp
        else:
            wf = np.concatenate((wf,temp),0)
        n=n+1
        if len(img)>numphase:
            for i in np.arange(numphase):
                img = np.delete(img,0,axis=0)
        else:
            break
    #WF_demo = WF[5,:,:]
    #reself.WFdemo = WF_demo
    wf = wf
    return wf
def getallangleield(img,numangle):
    n = 0
    phase =np.arange(numangle)
    while len(img)>0:
        print(len(img))
        temp = np.sum(img[phase,:,:],0)
        # st()
        temp = temp[np.newaxis,:,:]
        if n ==0:
            wf = temp
        else:
            wf = np.concatenate((wf,temp),0)
        n=n+1
        if len(img)>numangle:
            for i in np.arange(numangle):
                img = np.delete(img,0,axis=0)
        else:
            break
    
    #WF_demo = WF[5,:,:]
    #reself.WFdemo = WF_demo
    # st()
    wf = wf.reshape(-1,3,wf.shape[1],wf.shape[2]).mean(1)
    # wf = wf
    return wf
def getRMSrecon(img,numphase):
    # start = time.time()
    n = 0
    phase =np.arange(numphase)
    while len(img)>0:
        print(len(img))
        test = img[phase,:,:]*1.0
        sectioned=0
        for i in np.arange(numphase):
            temp = (test[1]-test[0])**2
            sectioned = sectioned+temp
            test=np.roll(test,1,0)
        sectioned = (sectioned)**.5
        sectioned = sectioned[np.newaxis,:,:]
        if n ==0:
            rms = sectioned
        else:
            rms = np.concatenate((rms,sectioned),0)
        n=n+1
        if len(img)>numphase:
            for i in np.arange(numphase):
                img = np.delete(img,0,axis=0)
        else:
            break
    return rms

def getSRSIM(img,numphase,x1,x2,x3):
    n = 0
    imgnum =np.arange(numphase*numangle)
    nz,nx,ny = np.shape(img)
    while len(img)>0:
        print(len(img))
        temp = img[imgnum,:,:]*1.0   
        # print(np.shape(temp))
        p = si.si2D(temp,numangle,numphase,wl,na)
        p.mu = 0.3
        p.fwhm = 0.99
        p.strength = 0.03
        p.minv = 0.
        p.eta = 0.06
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
        p.recon1(numangle,[x1[0],x2[0],x3[0]],[x1[1],x2[1],x3[1]],[-a1[1],-a2[1],-a3[1]],[a1[0],a2[0],a3[0]])
        test = p.finalimage.copy()
        test = test[np.newaxis,:,:]



        if n ==0:
            simimg = test
        else:
            simimg = np.concatenate((simimg,test),0)

        n=n+1

        if len(img)>(numphase*numangle):
            for i in imgnum:
                # print(imgnum)
                img = np.delete(img,0,axis=0)
        else:
            break
    return simimg

def getOSIM(img):
    n = 0
    imgnum =np.arange(numphase*numangle)
    nz,nx,ny = np.shape(img)
    while len(img)>0:
        print(len(img))
        temp = img[imgnum,:,:]*1.0    
        p = si.si2D(temp,numangle,numphase,wl,na)
        p.separate(0)
        test = np.fft.fftshift(p.separr[2])+np.fft.fftshift(p.separr[1])
        p.separate(1)
        test1 = np.fft.fftshift(p.separr[2])+np.fft.fftshift(p.separr[1])
        p.separate(2)
        test2 = np.fft.fftshift(p.separr[2])+np.fft.fftshift(p.separr[1])
        test = np.abs(test[np.newaxis,:,:])
        test1 = np.abs(test1[np.newaxis,:,:])
        test2 = np.abs(test2[np.newaxis,:,:])

        if n ==0:
            simimg = np.concatenate((test,test1,test2),0)
        else:
            simimg = np.concatenate((simimg,test,test1,test2),0)
        n=n+1
        if len(img)>(numphase*numangle):
            for i in imgnum:
                img = np.delete(img,0,axis=0)
        else:
            break
    return simimg



def window(eta):
    nx = 2048
    # nx = nx * 2
    #ny = self.ny * 2
    wxy = signal.tukey(nx, alpha=eta, sym=True)
    wx = np.tile(wxy,(nx,1))
    wy = wx.swapaxes(0,1)
    w = wx * wy
    return w


    
def driftcorrectafterrecon(img,shiftstep):
    n=0
    while len(img)>0:
        print(len(img))
        if n ==0:
            temp = img[0]
            final = temp[np.newaxis,:,:]
        else:
            temp = img[0]
            # temp = np.roll(temp,shift*n,1)
            temp = shift(temp,shift=(0,shiftstep)*n,mode='constant')
            temp = temp[np.newaxis,:,:]
            final = np.concatenate((final,temp),0)
        if len(img)>numangle:
            for i in np.arange(numangle):
                img = np.delete(img,0,axis=0)
            n=n+1
        else:
            break

    return final
        



def main():

    MODE = 'CALCULATE' # 'CALCULATE', 'RECONSTRUCT'

    # fns = r'H:/Onedrive for work/OneDrive - University of Georgia/Yang thesis/ReconSIMcode_04062022/2dsimrecon/beads_yg488_6um_300nmstep00_small_cut.tif'
    # fns = '../../data/20230118/beads_yg488_13um_500nmstep_20px_sin_512512_10.tif'
    # fns = '../../data/20230211/beads_yg488_10um_500nmstep_9px_sin_512512.tif'
    fns = '../../../mingws/sim_recon/sim_si2d_1.tif'
    filename = fns.split('/')[-1]
    path = fns.replace(fns.split('/')[-1],'')
    # st()
    img = tf.imread(fns)
    # pdb.set_trace()
    # wf = getWField(img,numphase)
    # save_path = os.path.join(path,'allangle_'+filename)
    # allangle = getallangleield(img,numangle)    
    # # # wf_test = driftcorrectafterrecon(wf,-0.5)
    # tf.imsave(save_path,allangle.astype(np.float32),photometric='minisblack')
    # st()
    # fns = r'H:/Onedrive for work/OneDrive - University of Georgia/Yang thesis/Thesis Writing/Yang/SIMLSFM/fly_LSFM_SIM_200ms_run2_50micron_500nmstep_run_2_0.tif'
    # img = tf.imread(fns)

    # rms = getRMSrecon(img, numphase)
    # tf.imsave('C:/Users/yang1990/Documents/Data/20220607/final_rms_image_6um__stack0.tif',rms.astype(np.float32),photometric='minisblack')

    # os = getOSIM(img)
    # tf.imsave('H:/Onedrive for work/OneDrive - University of Georgia/Yang thesis/Thesis Writing/Yang/SIMLSFM/fly_LSFM_SIM_200ms_run2_50micron_500nmstep_run_0_OS.tif',os.astype(np.float32),photometric='minisblack')

    # '''recon SIM'''
    if MODE == 'CALCULATE':
        test = img[0:numangle*numphase]
        p = si.si2D(test,numangle,numphase,wl,na)

        # st()


        p.mu = 0.1
        p.fwhm = 0.99
        p.strength = 0.03
        p.minv = 0.
        p.eta = 0.06
        p.separate(0)
        p.shift0()
        x1 = p.mapoverlap(angle1, patternspacing1, nps=8, r_ang=0.05, r_sp=0.04)
        # x1 = p.mapoverlap(x1[0], x1[1], nps=10, r_ang=0.05, r_sp=0.005)
        # x1 = p.mapoverlap(x1[0], x1[1], nps=10, r_ang=0.001, r_sp=0.001)
        
        p.separate(1)
        p.shift0()
        x2 = p.mapoverlap(angle2, patternspacing2, nps=8, r_ang=0.05, r_sp=0.04)
        # x2 = p.mapoverlap(x2[0], x2[1], nps=10, r_ang=0.05, r_sp=0.005)
        # x2 = p.mapoverlap(x2[0], x2[1], nps=10, r_ang=0.001, r_sp=0.001)
        
        p.separate(2)
        p.shift0()
        x3 = p.mapoverlap(angle3, patternspacing3, nps=8, r_ang=0.05, r_sp=0.04)
        # x3 = p.mapoverlap(x3[0], x3[1], nps=10, r_ang=0.05, r_sp=0.005)
        # x3 = p.mapoverlap(x3[0], x3[1], nps=10, r_ang=0.001, r_sp=0.001)

        print(x1)
        print(x2)
        print(x3)


    # elif MODE == "RECONSTRUCT":
    #     x1 = (-1.5243951476097255, 0.6867847296078067, 0.014507401110210953)
    #     x2 = (2.7038953007050175, 0.6918158179169475, 0.038245122366943135)
    #     x3 = (0.47571382603501566, 0.683000578060385, 0.014348449718422569)

        # x1 = [0.459, 0.453, 0.374]
        # x2 = [-2.80, 0.466, 0.721]
        # x3 = [2.87, 0.461, 0.451]
        # print(x1)
        # print(x2)
        # print(x3)

        print('Reconstruct for Wide Field All Angle')
        wf_all_angle_save_path = os.path.join(path,'allangle_'+filename)
        allangle = getallangleield(img,numangle)    
        tf.imsave(wf_all_angle_save_path,allangle.astype(np.float32),photometric='minisblack')

        # # wf_test = driftcorrectafterrecon(wf,-0.5)


        print('Reconstruct for SIM')
        sim_save_path = os.path.join(path,'final_simimg_'+filename)
        # st()
        simimg = getSRSIM(img,numphase,x1,x2,x3)
        
        # st()
        tf.imsave(sim_save_path,abs(simimg).astype(np.float32),photometric='minisblack')
        # tf.imsave('effective_OTF.tif',np.abs(np.fft.fftshift(p.Snum/p.Sden)).astype(np.float32),photometric='minisblack')

    
        # Code to be timed here
        end_time = time.time()
        duration = end_time - start_time
        print("Time taken: {:.6f} seconds".format(duration))

















if __name__ == '__main__':
    main()