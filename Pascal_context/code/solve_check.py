import caffe
import score as score
#import sampling_feature as save_f
import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

#weights = '/home/mipal/Data/HYOJIN/DATA/caffe_model/best_model_pixelnet.caffemodel'
#weights ='D:/Code_room/Caffe/caffe_Codes/HYOJIN_lab/best_model_pixelnet.caffemodel'
#weights = '/home/mipal/Data/HYOJIN/Pascal_Context59/Exp33_Pym2/snapshot/train_iter_24000.caffemodel'
weights ='backup/train_iter_8000.caffemodel'
# init
caffe.set_device(2)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
#A= solver.net.params.keys()
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/home/mipal/Data/HYOJIN/DATA/Pascal_Context_448_59cls/val.txt', dtype=str)
#val = np.loadtxt('D:/Code_room/DATA/VOC2010/VOC2010/Pascal_Context_224_59cls/val.txt', dtype=str)

#save_f.save_feature(solver, 'Result' , train, 'fc7', 'score_pascon', 're_sampling_label')
#print('end')
score.seg_tests(solver, False, val, layer='score_pascon')
#for i in range(5000):
    #solver.step(4000)
    #if((i %5)==0) :
        #print( str((i+1)*4000) + 'th seg test is saved ---------------')
        #score.seg_tests(solver, 'Result', val, layer='score_pascon')
    #else :
        #print( str((i+1)*4000) + 'th seg test is not saved ---------------')
        #score.seg_tests(solver, False, val, layer='score_pascon')
