import caffe

import numpy as np
from PIL import Image
import scipy.io
import random
from config import cfg
import copy


class PascalContextDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from Pascal context
    one-at-a-time while reshaping the net to preserve dimensions.

    This code follow this paper 
    
    Long, Jonathan, Evan Shelhamer, and Trevor Darrell. 
    "Fully convolutional networks for semantic segmentation." 
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.

       

    with 255 as the void label 

  
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - pascon_dir: path to pascal context dir
        - split: train / val / test
        - tops: list of tops to output from {color, depth, hha, label}
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for pascal context 33class semantic segmentation.

       
        """
        # config
        params = eval(self.param_str)
        self.pascon_dir = params['pascon_dir']
        self.split = params['split']
        self.tops = params['tops']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.batch = params.get('batch',1)
        cfg.batch = self.batch

        # store top data for reshape + forward
        self.data = {}

        # means
        self.mean_bgr = np.array((104.007, 116.669, 122.679), dtype=np.float32)
        

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.pascon_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.temp_idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        self.idx = np.zeros(self.batch)
        #random.seed(self.seed)
        for i in range(0,self.batch):
            if self.random:               
                self.temp_idx = random.randint(0, len(self.indices)-1)
                self.idx[i] = self.temp_idx
            else:    
                self.idx[i] = self.temp_idx            
                self.temp_idx +=1
      

    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        
        for i, t in enumerate(self.tops): # img(batch*3*h*w), label(batch*1*h*w) superpixel(batch*1*h*w) N_sp(batch*1)
            self.data[t] = np.array([])
            for j in range(0,self.batch):
                temp = self.load(t, self.indices[int(self.idx[j])])
                temp =temp.reshape(1,*temp.shape)  # c*h*w -> 1*c*h*w
                self.data[t] = np.vstack([self.data[t], temp]) if self.data[t].size else temp
               
            if((t=='N_pixel') & (self.batch==1)):
                self.data[t] = self.data[t][np.newaxis,...]
                
            if(t=='N_pixel'):
                cfg.N_pixel = copy.deepcopy(self.data[t])
                
            #C=self.data[t].shape
            top[i].reshape(*self.data[t].shape)
           
    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]
            
        name_d=[]
        for i in range(0,self.batch):
            name_d.append(self.indices[int(self.idx[i])])
        cfg.data_name = copy.deepcopy(name_d)
        cfg.data_idx = copy.deepcopy(self.idx)
       
    
        # pick next input
        
        for i in range(0,self.batch):
            if self.random:
                A = random.randint(0, len(self.indices)-1)
                B=len(self.indices)
                #self.temp_idx = random.randint(0, len(self.indices)-1)
                self.idx[i] = A #self.temp_idx
            else:
                if self.temp_idx == len(self.indices):
                    self.temp_idx = 0                  
                self.idx[i] = self.temp_idx
                self.temp_idx +=1
                              
      
        #if self.random:
            #self.idx = random.randint(0, len(self.indices)-1)
        #else:
            #self.idx += 1
            #if self.idx == len(self.indices):
                #self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, top, idx):
        if top == 'color':
            return self.load_image(idx)
        elif top == 'label':
            return self.load_label(idx)
        elif top =='superpixel':
            return self.load_superpixel(idx)
        elif top == 'N_pixel':
            return self.load_Npixel(idx)
        elif top =='sc_label':
            return self.load_sc_label(idx)
  
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/images/{}.jpg'.format(self.pascon_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]  
        in_ -= self.mean_bgr
        in_ = in_.transpose((2,0,1))
        
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 1-150 =>0~149 and void is 255=>254 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        label = scipy.io.loadmat('{}/label_mat/{}.mat'.format(self.pascon_dir, idx))['resize_label'].astype(np.uint8)
        #label -= 1  # rotate labels
        label = label[np.newaxis, ...]
        org_shape = label.shape # 1* H W
        cfg.height = org_shape[1]
        cfg.width = org_shape[2]        
        return label
    
    def load_sc_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 1-150 =>0~149 and void is 255=>254 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        sc_label = scipy.io.loadmat('{}/superset/data6/{}.mat'.format(self.pascon_dir, idx))['sc_label'].astype(np.uint8)
        #label -= 1  # rotate labels
        sc_label = sc_label[np.newaxis, ...]
         
        return sc_label    
     
    def load_superpixel(self, idx):
        
        superpixel = scipy.io.loadmat('{}/superpixel_200/{}.mat'.format(self.pascon_dir, idx))['this_str']['L']
        A=superpixel.shape
        superpixel = superpixel[0,0]
        B=superpixel.shape
        superpixel = superpixel[np.newaxis, ...]
       
        return superpixel
    
    def load_Npixel(self, idx): # get each superpixel number
        N_pixel = scipy.io.loadmat('{}/superpixel_200/{}.mat'.format(self.pascon_dir, idx))['this_str']['NumLabels']
        A=N_pixel[0,0]
        N_pixel =A[0,0]
       
        return N_pixel
 
