import caffe
import os
import sys
import numpy as np
from numpy import array 
from PIL import Image
import scipy.io
from config import cfg
import copy


class RandSPTestSamplingLayer(caffe.Layer):
 
  
      def setup(self, bottom, top):
            
            #setup variable
            #bottom - superpixel(N*1*h*w), N_superpixel(N*1)
            #top - rand_coord
            params = eval(self.param_str)
           
            self.superpixel=None #superpixel result N*1*h*w
            self.N_sp = 0 # number of superpixel (N*1) 
            self.N_sample= params['N_sample']        
           
        
            self.rand_idx=None # sampling sp order idx
            self.rand_coord=None # save sp rand_coord
            self.N_batch = None
            
            cfg.N_sample = self.N_sample
      
      
      def reshape(self, bottom, top):
            
            self.N_batch = bottom[0].data.shape[0]       
                            
            top[0].reshape(self.N_batch*self.N_sample*3) # batch, x(c), y(r)
          
            
      
      def forward(self, bottom, top):
           
            # take information
            self.superpixel = bottom[0].data; #batch * 1* H*W
            cfg.superpixel = self.superpixel
            
            self.N_sp = bottom[1].data.astype(np.uint32); #batch            
                              
          
            self.rand_idx = np.array([])
            self.rand_coord = np.array([], dtype=np.int32).reshape(0,1)
            #replace_idx = np.empty(shape=(self.N_batch,), dtype=object)
            #check_coord_r = np.zeros(self.N_batch)
            #check_coord_c = np.zeros(self.N_batch)            
            for batch in range(0,self.N_batch): #  each batch
                 
                  this_N_sp = self.N_sp[batch]
                  
                  temp_rand_idx = np.linspace(1,this_N_sp, this_N_sp)
                  temp_zeros = np.zeros( (self.N_sample - this_N_sp))
                  temp_rand_idx = np.concatenate((temp_rand_idx, temp_zeros))
               
                  
                  #replace_idx[batch] = temp_replace
                  this_sp = self.superpixel[batch,:,:,:] 
             
                  #if(self.if_rand):
                        #np.random.shuffle(temp_rand_idx)
                
                  self.rand_idx=np.vstack([self.rand_idx,temp_rand_idx]) if self.rand_idx.size else temp_rand_idx # save rand_sp_index table
                  # temp_rand_idx (N_sample) => rand_idx (batch * N_sample)
                  
                  for i in range(0,this_N_sp): ############ for - N_sample
                        this_idx = temp_rand_idx[i];
                        Mask_idx = (this_sp[0]==this_idx)
                        
                       
                        # make this sp center coord for test
                        row_mask = np.where(Mask_idx==1)[0]
                        col_mask = np.where(Mask_idx==1)[1]
                        
                       
                        if((row_mask.size != np.sum(Mask_idx)) | (col_mask.size != np.sum(Mask_idx))):
                              print('You kill me!!!! row_mask or col mask size not equal to mask_idx')
                              raise NotImplementedError  
                    
                        mean_row = np.round(np.mean(row_mask))
                        mean_col = np.round(np.mean(col_mask)) 
                        
                        self.rand_coord=np.vstack([self.rand_coord,int(batch)]) # batch
                        self.rand_coord=np.vstack([self.rand_coord,int(mean_col)]) # x_pt
                        self.rand_coord=np.vstack([self.rand_coord,int(mean_row)]) # y_pt
                        
                        #if (this_idx == 100):
                              #check_coord_r[batch] = int(rand_row)  
                              #check_coord_c[batch] = int(rand_col)
                  for i in range(this_N_sp, self.N_sample):
                        self.rand_coord=np.vstack([self.rand_coord,int(batch)]) # batch
                        self.rand_coord=np.vstack([self.rand_coord,int(1)]) # x_pt
                        self.rand_coord=np.vstack([self.rand_coord,int(1)]) # y_pt
                        
                  ####### end - N_sample
                  #A= temp_rand_coord.shape
                 
                  
            ######end -batch
          #  print(self.rand_coord)
      
            top[0].data[...] = self.rand_coord.reshape(self.rand_coord.shape[0])
            #print(check_coord_r)
            #print(check_coord_c)
           
            cfg.rand_idx = self.rand_idx
            cfg.rand_coord = self.rand_coord
           
              
      def backward(self, top, propagate_down, bottom): 
            pass
     