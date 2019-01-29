from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
from config import cfg
import time

def fast_hist(a, b, n):
    A_=a.shape
    B_=b.shape
    k = (a >= 0) & (a < n)
    K_=k.shape
  
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score_pascon', gt='label'):
    n_cl = net.blobs[layer].channels
    
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    dataID=0
    total_time=0;
    task =0
    limit_T = 10    
    while dataID < len(dataset):
	task=task+1
        start_time = time.time()        
        net.forward()
        total_time = total_time + (time.time() - start_time)
        if (task*5) ==limit_T:
            print(total_time)
            print(total_time/limit_T)
            break   
        for batch in range(0,cfg.batch):
                     
            # get each batch sampling img
            # net.blobs[layer].shape => N*class_num*S 
            start_time2 = time.time()
            sampling_img = net.blobs[layer].data[batch].argmax(0) # cls*S -> S
        
            
           
            #get transfrom info
            this_SPidx = cfg.rand_idx[batch,:]
            this_sp = cfg.superpixel[batch,:,:,:] 
            
            #A=this_replace.shape
            #from first to end pixel find sp 
            org_img =  np.zeros(net.blobs[gt].data[0,0].shape) # org h*w
            
           
            this_N_sp = cfg.N_pixel[batch]
 	   
            for i in range(0, this_N_sp):
                this_idx =this_SPidx[i]
                mask_id = (this_sp[0] == this_idx) 
                class_id = sampling_img[i]            
                org_img += class_id*mask_id
   
            
    	    org_img=org_img.astype(np.int64)
	    total_time = total_time + (time.time() - start_time2)
	    if (task*5) == limit_T:
		print(total_time)
		print(total_time/limit_T)
		break 
            if save_dir:
		    im = Image.fromarray(org_img.astype(np.uint8), mode='P')              
		    im.save(os.path.join(save_dir, dataset[cfg.data_idx[batch]] + '.png'))   
     
            #######
            
            #C=net.blobs[gt].data[batch,0] 
            #C_s=net.blobs[gt].data[batch,0].shape
            #D=net.blobs[gt].data
            #D_s = net.blobs[gt].data.shape
            #B=org_img.flatten().shape
            hist += fast_hist(net.blobs[gt].data[batch, 0].flatten(),
                                    org_img.flatten(),
                                    n_cl)
          
           
            
            dataID+=1
            # compute the loss as well
       
        loss += net.blobs['loss_pascon'].data.flat[0] # add all batch
        ##########end for batch
    ####### end while len(dataset)
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score_pascon', gt='label'):
    print '>>>', datetime.now(), 'Begin seg tests'
    solver.test_nets[0].share_with(solver.net)
    do_seg_tests(solver.test_nets[0], solver.iter, save_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score_pascon', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    print '>>>', datetime.now(), 'Iteration', iter, 'mean AP', np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    freq = hist.sum(1) / hist.sum()
    print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
            (freq[freq > 0] * iu[freq > 0]).sum()
    return hist
