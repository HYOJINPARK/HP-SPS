import caffe

import numpy as np
from PIL import Image
import scipy.io
import random
from config import cfg
# version 1_30
class ReshapeDataLayer(caffe.Layer):
    
     def setup(self, bottom, top):
          # score : (N_sampling*batch) x (class_c)   => batch * class_C * N_sampling
          # label : (N_sampling*batch) x (1) => batch * 1* N_sampling
          # bottom[0] : score 
          # bottom[1] : label
    
          self.score=None
          self.label=None
          self.rand_idx =None
          self.batch = cfg.batch
          params = eval(self.param_str)
          self.Test = params.get('Test', False)
          self.sample_n = params.get('N_sample',1500)  
         
          self.n_top = params.get('N_label', 1)
          self.class_num = bottom[0].shape[1]
          self.rand_point = params.get('Get_rand', False)
     
     def reshape(self, bottom, top):
          # top[0] = reshaped score
          # top[1] = reshaped label
         

          if(self.rand_point):
               self.rand_idx = bottom[2]
               cfg.rand_idx = self.rand_idx               
          
          top[0].reshape(self.batch, self.class_num, self.sample_n);
          for i in range(1,self.n_top +1):
               top[i].reshape(self.batch, 1, self.sample_n);          
     
     def forward(self, bottom, top):
          start_id = 0
          end_id=0
          #A=bottom[1].shape
          #B=bottom[1].data[0:1500,:]
          #C=B.shape
          #print(bottom[0].data.shape) #3000*150
          #print(bottom[1].data.shape) #3000*3
          for i in range(0,self.batch):
               end_id = start_id + self.sample_n
               
               this_t0 = np.zeros([self.class_num, self.sample_n]) #150 X 1500
               this_b0 = bottom[0].data[start_id:end_id,:]
              
               for j in range(0,self.class_num): # 150*1500
                                   this_t0[j,:] = this_b0[:,j]               
               
               top[0].data[i,...] = this_t0
                
               for k in range(1, self.n_top+1):
                    this_t1 = np.zeros([1,self.sample_n])
                    this_b1 = bottom[k].data[start_id:end_id,:] # get each batch data point
                    this_t1[0,:] = this_b1[:,0]
                    top[k].data[i,...] = this_t1
                  
               
               start_id = end_id  
          #end each batch
     
     
          
          #B=top[1].data
          #print(B)
          #A=top[0].data
               
     def backward(self, top, propagate_down, bottom):
          top_diff = top[0].diff[...]
          btm_diff = np.array([])
          if(propagate_down[0]):          
               for i in range(0,self.batch):
                    this_t0 = top_diff[i,...] # 150X1500
                    this_b0 = np.zeros([self.sample_n, self.class_num]) # 1500 X 150                    
                    for j in range(0,self.class_num): 
                         this_b0[:,j] = this_t0[j,:]
                    #end class_num*sample_n
                    btm_diff = np.vstack([btm_diff,this_b0]) if btm_diff.size else this_b0 # 3000 X 150
                   
          #end each batch
          #A=btm_diff.shape
          bottom[0].diff[...] = btm_diff