import caffe
import score as score
import matplotlib.pyplot as mp
from matplotlib import pyplot as plt
import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

#weights = '/home/mipal/Data/HYOJIN/DATA/caffe_model/best_model_pixelnet.caffemodel'
#weights ='D:/Code_room/Caffe/caffe_Codes/HYOJIN_lab/best_model_pixelnet.caffemodel'
#weights = '/home/mipal/Data/HYOJIN/Pascal_Context59/Exp33_Pym2/snapshot/train_iter_24000.caffemodel'
weights ='snapshot/train_iter_16000.caffemodel'
# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver_plot.prototxt')
solver.net.copy_from(weights)

# surgeries
#A= solver.net.params.keys()
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)

# scoring
#val = np.loadtxt('/home/mipal/HYOJIN/DATA/Pascal_Context_448_59cls/val_plot.txt', dtype=str)
val = np.loadtxt('D:/Code_room/DATA/VOC2010/VOC2010/Pascal_Context_448_59cls/val_plot.txt', dtype=str)

bins=100
N_batch=5
for_w = solver.net.forward()
back_w = solver.net.backward()

print('figure information')
net_name = np.array(['score_pascon','fc7', 'fc6'])

sigma_p = np.array([ 9.40706866,  0.62157951,  0.95176241])

#############################################################################
feature_diff = np.array([])
abs_diff = np.array([])
diff_std = np.array([])
diff_gap = np.array([])
diff_mean = np.array([])
diff_Num = np.array([])
for i in range(net_name.shape[0]):
    Num= solver.net.blobs[net_name[i]].diff.shape
    temp_diff=np.zeros(Num[1])
    temp_abs=np.zeros(Num[1])
    if i==0:
        diff_Num = np.append(diff_Num, Num[1])
    else:
        diff_Num = np.append(diff_Num, diff_Num[i-1]+Num[1])
        
    for batch in range(N_batch):
        diff_score = solver.net.blobs[net_name[i]].diff
        this_diff = diff_score[batch,:,:,0] # dim * n_sp
        sum_diff = np.sum(this_diff,1)
        abs_sum = np.sum(np.abs(this_diff),1) 

        temp_diff = temp_diff + sum_diff
        temp_abs = temp_abs + abs_sum
    print(temp_diff[10:30])    
    diff_std= np.append(diff_std, np.std(temp_abs))      
    diff_gap = np.append(diff_gap, max(temp_abs) - min(temp_abs))
    diff_mean = np.append(diff_mean, np.mean(temp_abs))
    feature_diff = np.append(feature_diff, temp_diff)
    abs_diff = np.append(abs_diff, temp_abs)
      
      
target_diff = abs_diff[60::]
norm_gap = (diff_gap - diff_mean) / diff_std
print('diff std : ')
print( np.std(target_diff))
print(diff_std)     
print(diff_gap)
print(norm_gap)
print(diff_mean)



'''
diff std : 
1.60268254582
[ 7.97711395  0.83749407  1.26906835]
[ 39.86239659   7.44450295   8.50643313]
[ 4.3167665   3.78901064  5.21034426]

diff std : 
1.57189351522
[ 8.07970564  0.8295289   1.23303953]
[ 32.93706905   7.25973481   8.15297518]
[ 3.40415457  3.63810252  5.06811184]
#'''
#fig_info = mp.figure(0)
X=np.array(range(feature_diff.shape[0]))
X=X+1
#A=max(abs_diff)
#B=min(abs_diff)
#print('max :  ', A , 'min :   ', B)
#mp.xlim([-2,feature_diff.shape[0]+1])
#mp.ylim([B-0.01, A+0.01])
#mp.title('abs_diff')
#mp.xlabel('feature')
#mp.ylabel('abs_value')    
#mp.bar(X,abs_diff)

fig_info = mp.figure(3)
mp.xlim([-2,feature_diff.shape[0]+1])
mp.ylim([-0.01,12])
mp.title('abs_diff_ylim')
mp.xlabel('feature')
mp.ylabel('abs_value')    
mp.bar(X,abs_diff)    
start_p = 0
end_p = diff_Num[0]
const = 6;
for i in range(diff_Num.shape[0]):
    lower_s , lower_e = [start_p, end_p ] , [diff_mean[i] - const*sigma_p[i], diff_mean[i] - const*sigma_p[i]]
    
    
    upper_s , upper_e= [start_p, end_p], [diff_mean[i] + const*sigma_p[i], diff_mean[i] + const*sigma_p[i]]
    print(lower_s, lower_e)
    print(upper_s, upper_e)
    plt.plot(upper_s, upper_e, color = 'red' )
    plt.plot(lower_s, lower_e, color = 'red' )
    if (i!=(diff_Num.shape[0]-1)):
        start_p = diff_Num[i]
        end_p = diff_Num[i+1]
#fig_info = mp.figure(2)

#A=max(feature_diff)
#B=min(feature_diff)
#print('max :  ', A , 'min :   ', B)
#mp.xlim([-2,feature_diff.shape[0]+1])
#mp.ylim([B-0.01, A+0.01])
##mp.ylim([-10,10])
#mp.title('sum_diff_ylim')
#mp.xlabel('feature')
#mp.ylabel('sum_value')    
#mp.bar(X,feature_diff)    
    
mp.show()       


 
############################################################################################
#for batch in range(2):
    #org = mp.figure(batch*3 +1)
    #non_0 = mp.figure(batch*3 + 2)
    #plot_t = plt.figure(batch*3 +3)
    #abs_plot = plt.figure(batch*3 + 4)
#for batch in range(2):
   
    #feature_diff = np.array([])
    #abs_diff = np.array([])
    #for i in range(net_name.shape[0]):
        #diff_score = solver.net.blobs[net_name[i]].diff
        #this_diff = diff_score[batch,:,:,0] # dim * n_sp
        #sum_diff = np.sum(this_diff,1)
        #abs_sum = np.sum(np.abs(this_diff),1)
        #feature_diff = np.append(feature_diff, sum_diff)
        #abs_diff = np.append(abs_diff, abs_sum)
        #this_diff = this_diff.flatten()
        #mp.figure(batch*3+1)
        #mp.subplot(4,2,i+1) 
        #mp.style.use('ggplot')
        #mp.title(net_name[i])
        #mp.xlabel('bin')
        #mp.ylabel('freq')
        #info = mp.hist(this_diff,bins)
        #freq_a = info[0]
        ##bin_a = info[1]
        #max_f = freq_a.max()
        ##idx=np.where(max_f==freq_a)
        ##print( 'this max freq bin value ', bin_a[idx],' ~ ', bin_a[np.array(idx)+1][0])
        
        #mp.figure(batch*3+2)
      
        #mp.subplot(4,2,i+1) 
        #mp.style.use('ggplot')
        #mp.title(net_name[i])
        #mp.xlabel('bin')
        #mp.ylabel('freq')
        #mp.ylim([0,int(max_f/10)])
        #mp.hist(this_diff,bins)
    
    
    #fig_info = mp.figure(batch*3+3)
    #X=np.array(range(feature_diff.shape[0]))
    #X=X+1
    #A=max(feature_diff)
    #B=min(feature_diff)
    #print('max :  ', A , 'min :   ', B)
    #mp.xlim([-2,feature_diff.shape[0]+1])
    #mp.ylim([B-0.01, A+0.01])
    #mp.bar(X,feature_diff, color='red')

    
    #fig_info = mp.figure(batch*3+4)
    #X=np.array(range(feature_diff.shape[0]))
    #X=X+1
    #A=max(abs_diff)
    #B=min(abs_diff)
    #print('max :  ', A , 'min :   ', B)
    #mp.xlim([-2,feature_diff.shape[0]+1])
    #mp.ylim([B-0.01, A+0.01])
    #mp.bar(X,feature_diff, color='red')
    
#mp.show()
 
