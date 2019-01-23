import numpy as np
import scipy.io as sio
from tqdm import tqdm
import time
import sys

I_VAL = 1
FEATURE_PATH = './features_paviau/feature_%d.npy'%(I_VAL)
# feature with shape: Channel, height, width

KER_PATH = 'ker_%d.txt'%(I_VAL)

COORDS_PATH = './Pavia_University/paviau_coord_%d.txt'%(I_VAL)
CENTER_LIST_PATH = './features_paviau/center_list_%d.npy'%(I_VAL) 

DATA_GT_PATH = './Pavia_University/PaviaU_gt.mat'
data_mat_gt = sio.loadmat(DATA_GT_PATH)['paviaU_gt']

features = np.load(FEATURE_PATH)

center_list = np.load(CENTER_LIST_PATH)


shape = features.shape[1:]

mask = np.ones(shape)
with open(COORDS_PATH,'r') as tr:
    train_list = tr.readlines()

data_list = {}
train_set = set()
for item in train_list:
    idx_i, idx_j = eval(item)
    mask[idx_i,idx_j] = 0
    if not item in train_set:
        train_set.add(item)

def in_train_data(idx_i, idx_j):
    if str((idx_i, idx_j))+'\n' in train_set :
        return True

#### load the kernels
print('loading the kernels...')
kernels = dict()
with open(KER_PATH,'r') as f:
    kernel_data = f.readlines()

buffers = []
counter = 0
for item in kernel_data:
    if item[0]=='_':
        counter += 1
        buffers.append(item[1:].split('_'))
    else:
        score_data = np.array(eval(item))
        start = 0
        for coords in buffers:
            bias = eval(coords[1])[0] * eval(coords[1])[1]
            kernels[eval(coords[0])] = score_data[start:start+bias].reshape(eval(coords[1]))
            start += bias
        assert start == score_data.shape[0]
        buffers = []

####### Testing
correct = 0
fail = 0
fail_pair = []
correct_dict = dict()
predict_dict = [0]*10
groundt_dict = [0]*10
result = []
kernel = 10
real_kernel = 10
for idx_i in tqdm(range(shape[0])):
    for idx_j in range(shape[1]):
        if data_mat_gt[idx_i,idx_j] == 0:
            continue
        if in_train_data(idx_i,idx_j):
            continue
            
        predict_label = []

        coord_rel = (idx_i - max(idx_i-kernel+1,0), idx_j - max(idx_j-kernel+1,0))
        
        ker_shape = kernels[(idx_i,idx_j)].shape
        
        new_kernel = kernels[(idx_i,idx_j)]\
        [max(coord_rel[0]-real_kernel+1,0):min(coord_rel[0]+real_kernel,ker_shape[0]),\
        max(coord_rel[1]-real_kernel+1,0):min(coord_rel[1]+real_kernel,ker_shape[1])]
        
        weights = (new_kernel.reshape(-1)>0.01)\
                *mask[max(idx_i-real_kernel+1,0):min(idx_i+real_kernel,shape[0]),\
                   max(idx_j-real_kernel+1,0):min(idx_j+real_kernel,shape[1])].reshape(-1)
            
        if np.sum(weights) == 0:
            weights = (new_kernel.reshape(-1)>=0)\
                *mask[max(idx_i-real_kernel+1,0):min(idx_i+real_kernel,shape[0]),\
                   max(idx_j-real_kernel+1,0):min(idx_j+real_kernel,shape[1])].reshape(-1)
                
        av_feature = \
                np.average(features[:,max(idx_i-real_kernel+1,0):min(idx_i+real_kernel,shape[0]),\
                   max(idx_j-real_kernel+1,0):min(idx_j+real_kernel,shape[1])].reshape((32,-1)),axis=1,\
                   weights=weights) 
        dist = 9999999999999999
        label_av = -1
        for idx, center in enumerate(center_list):
            new_dist = np.sum((av_feature - center)**2)#/np.sum((center)**2)
                #new_dist = scipy.spatial.distance.cosine(av_feature, center)
            if dist > new_dist:
                dist = new_dist
                label_av = idx
        label_av += 1
        
        if data_mat_gt[idx_i,idx_j] != 0:
            
            predict_dict[label_av] += 1
            groundt_dict[data_mat_gt[idx_i,idx_j]] += 1
            
            if not data_mat_gt[idx_i,idx_j] in correct_dict:
                correct_dict[data_mat_gt[idx_i,idx_j]] = [0,0]
            if label_av == data_mat_gt[idx_i,idx_j]:
                correct += 1
                correct_dict[data_mat_gt[idx_i,idx_j]][0] += 1
                correct_dict[data_mat_gt[idx_i,idx_j]][1] += 1
            else:
                fail += 1
                fail_pair.append((data_mat_gt[idx_i,idx_j], label_av))
                correct_dict[data_mat_gt[idx_i,idx_j]][1] += 1
sys.stdout.write('\n')
sum_correct = 0
for key in correct_dict:
    print('%2d, %5d, %5d, %.4f'%(key,correct_dict[key][0],\
                                correct_dict[key][1],\
                                correct_dict[key][0]*1./correct_dict[key][1]))
    sum_correct += correct_dict[key][0]*1./correct_dict[key][1]
    result.append(correct_dict[key][0]*1./correct_dict[key][1])
print(correct,fail)
oa = correct/(correct+fail*1.)
aa = sum_correct/9
pe = np.sum(np.array(predict_dict)*np.array(groundt_dict))*1./(np.sum(np.array(predict_dict))**2)
kc = (oa-pe)/(1-pe)
print('overall accuracy: %.4f'%(oa))
print('average accuracy: %.4f'%(aa))
print('kappa coefficien: %.4f'%(kc))
