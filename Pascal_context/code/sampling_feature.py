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
    n_class = 60
    dataID=0
    feature_set = np.array([],dtype=np.float32) #  dtype=np.int32
    label_set = np.array([], dtype=np.float32).reshape(0,1)
    each_max = 1000;
    limit_n = 10; # samling per each img
    limit_class = 3 # limit number of samplng class type
    
    curr_n = np.zeros(n_class) # current the number of sample 
    end_idx = np.zeros(n_class, dtype = np.bool)
   
    while dataID < len(dataset):
        
        net.forward()
        
        for batch in range(0,cfg.batch):
            dataID = dataID +1
            #current sample candidate 
            sampling_img = net.blobs[score_layer].data[batch].argmax(0) # cls*S -> S                  
            candidate = np.unique(sampling_img)
            this_f = net.blobs[feature_layer].data[batch]          
            this_l = net.blobs[gt_layer].data[0]            
            
            #and remove end_class idx if has 
            this_remove =np.where(end_idx == True)[0] # end class number
            #this_remove = this_remove.reshape(this_remove.shape[1])
            check1 = this_remove.shape
            remove_idx = np.array([]).reshape(0,1)
            for i in range(0,this_remove.shape[0]):
                A= np.where(candidate == this_remove[i])[0]
                remove_idx = np.vstack([remove_idx, int(A)])
            candidate = np.delete(candidate, remove_idx)
            
            if(candidate.shape[0] > limit_class):
                this_idx = np.random.choice(range(0,candidate.shape[0]), limit_class, replace = False)
                candidate = candidate[this_idx]
            
            #################  start mask for target sample
            
            # find estimation == label 
            gt = net.blobs[gt_layer].data[batch,0]
            gt = gt.reshape(gt.shape[0],1)
            True_mask = (sampling_img == gt)  #sample * 1      -> 2D
            True_mask = True_mask.reshape(True_mask.shape[0])   # sample -> 1D
            
            # find this class idx
            for i in range(0,candidate.shape[0]):
                this_class = candidate[i]
                this_mask = (gt ==this_class)
                this_mask = this_mask.reshape(this_mask.shape[0])
            
                # make target sample location 
                temp_target = ( True_mask & this_mask) # true positive this class
                save_target = np.zeros(temp_target.shape, dtype=np.bool) # 1D
                #but there is a limitation of limit_n so save idx.
                if(np.sum(temp_target)>0):
                    if(np.sum(temp_target) > limit_n):
                        mask_idx = np.where(temp_target==True)[0]             
                        rand_id = np.random.choice(range(0,mask_idx.shape[0]), limit_n, replace = False)
                        save_target[mask_idx[rand_id]] = True                
                    else:
                        save_target = temp_target
            
                
                    save_f = this_f[:,save_target,:]            
                    save_l = this_l[:,save_target]
                    
                    save_f= save_f.reshape(save_f.shape[0], save_f.shape[1])
                    save_f = save_f.transpose(1,0)
                    save_l = save_l.transpose(1,0)
                    
                    save_f.astype(np.float32)
                    save_l.astype(np.float32)
                 
                    
                    feature_set =np.vstack([feature_set,save_f]) if feature_set.size else save_f               
                    label_set = np.vstack([label_set, save_l])                 
                
                    # get sample and save current the number of sample
                    curr_n[this_class] = curr_n[this_class] + np.sum(save_target)
                    
                    # update complete_class idx
                    if (curr_n[this_class] > each_max-1) :
                        end_idx[this_class] = True   
                        print( 'extract  ' + str(this_class) + 'th class ') 
                        if(np.sum(end_idx) == n_class):
                            print(' end all sampling ' )
                            break
                
                
           
        ##########end for batch
    print ( ' we end ' + str(np.sum(end_idx)) + '  type class sampling' )
    if (np.sum(end_idx) < n_class) :
        print( ' print lack number of incomplete one')
        for i in range(0,n_class):
            if(end_idx[n_class] == False):
                print (str(i) + 'th sample number is ' + str(curr_n[i]) )
       
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

