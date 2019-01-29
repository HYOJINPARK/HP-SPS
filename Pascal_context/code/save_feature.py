from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from PIL import Image
from config import cfg
from scipy import io


def compute_feature(net, dataset, feature_layer, score_layer, gt_layer):
    
    dataID=0
    feature_set = np.array([],dtype=np.float32) #  dtype=np.int32
    label_set = np.array([], dtype=np.float32).reshape(0,1)
   
    while dataID < len(dataset):
        print('b4 net')
        net.forward()
        print('end net')
        for batch in range(0,cfg.batch):
           
            
            sampling_img = net.blobs[score_layer].data[batch].argmax(0) # cls*S -> S
                            
            # if estimation == label 
            gt = net.blobs[gt_layer].data[batch,0]
            gt = gt.reshape(gt.shape[0],1)
            save_target = (sampling_img == gt)        
            save_target = save_target.reshape(save_target.shape[0])
            #A1= save_target.shape
            N_get= np.sum(save_target)
            
            this_f = net.blobs[feature_layer].data[batch]          
            this_l = net.blobs[gt_layer].data[0]
            
            #A=this_f.shape
            #B=this_l.shape
            print('done this ')
            save_f = this_f[:,save_target,:]            
            save_l = this_l[:,save_target]
            
            save_f= save_f.reshape(save_f.shape[0], save_f.shape[1])
            save_f = save_f.transpose(1,0)
            save_l = save_l.transpose(1,0)
            
            save_f.astype(np.float32)
            save_l.astype(np.float32)
            print('done save')
                
            #C= save_f.shape
            #D=save_l.shape
            # save that feature and label
            feature_set =np.vstack([feature_set,save_f]) if feature_set.size else save_f
            #E=feature_set.shape
            print('done stack f')
            label_set = np.vstack([label_set, save_l]) 
            print('done stack l')
            #F=label_set.shape
    
            #######

            
            dataID+=1
            # compute the loss as well
            print(dataID)
            if ((1000< dataID) & (dataID < 1010)) :
                data = {'feature' : feature_set, 'label': label_set}
                io.savemat('feature_label_1.mat',data) 
                print('1000')
            elif((2000< dataID) & (dataID < 2010)) :
                data = {'feature' : feature_set, 'label': label_set}
                io.savemat('feature_label_2.mat',data) 
                print('2000')
            elif ((3000< dataID) & (dataID < 3010)) :
                data = {'feature' : feature_set, 'label': label_set}
                io.savemat('feature_label_3.mat',data) 
                print('3000')
            elif ((4000< dataID) & (dataID < 4010)) :
                data = {'feature' : feature_set, 'label': label_set}
                io.savemat('feature_label_4.mat',data) 
                print('4000')
                
       
        ##########end for batch
           
       
    ####### end while len(dataset)
    return feature_set, label_set

def save_feature(solver, save_format, dataset, feature_layer, score_layer, gt_layer):
    print '>>>', datetime.now(), 'Begin feature save'
    
    solver.test_nets[0].share_with(solver.net)
    (feature_set, label_set) = compute_feature(solver.test_nets[0],  dataset,feature_layer, score_layer, gt_layer)
    
    #if save_format:
        #save_format = save_format.format(solver.iter)    
    
    data = {'feature' : feature_set, 'label': label_set}
    io.savemat('feature_label.mat',data)
    print('done')

