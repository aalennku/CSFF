import scipy.io as sio
import numpy as np
import random
import itertools

I_VAL = 1
# I_VAL is used for indecating different rounds of the experiments.

DATA_PATH = './Pavia_University/PaviaU.mat'
DATA_GT_PATH = './Pavia_University/PaviaU_gt.mat'
COORDS_PATH = './Pavia_University/paviau_coord_%d.txt'%(I_VAL)
# COORDS_PATH is a text file records the coordinations of training samples with format '(x, y)' in each line for a training sample.

data_mat = sio.loadmat(DATA_PATH)['paviaU']#[:,:,:100]
data_mat_gt = sio.loadmat(DATA_GT_PATH)['paviaU_gt']

height, width, channel = data_mat.shape
data_mean, data_std = np.average(data_mat), np.var(data_mat)**0.5

coords = []
with open(COORDS_PATH,'r') as f:
    for item in f.readlines():
        coords.append(eval(item))

data_mat_dict = {}
for item in coords:
    if not data_mat_gt[item] in data_mat_dict:
        data_mat_dict[data_mat_gt[item]] = []
    data_mat_dict[data_mat_gt[item]].append((data_mat[item]-data_mean) / data_std)

data_train = []
data_label = []
pos_cnt = 0
neg_cnt = 0
for item_a, item_b in itertools.product(data_mat_dict.keys(), data_mat_dict.keys()):
    if item_a == item_b:
        data_train += list(itertools.product(data_mat_dict[item_a],data_mat_dict[item_b]))
        data_label += [1]*len(list(itertools.product(data_mat_dict[item_a],data_mat_dict[item_b])))
        pos_cnt += len(list(itertools.product(data_mat_dict[item_a],data_mat_dict[item_b])))
    if item_a < item_b:
        _data = list(itertools.product(data_mat_dict[item_a],data_mat_dict[item_b]))+\
        list(itertools.product(data_mat_dict[item_b],data_mat_dict[item_a]))
        random.shuffle(_data)
        data_train += _data[:len(_data)/2]
        data_label += [0]*len(_data[:len(_data)/2])
        neg_cnt += len(_data[:len(_data)/2])
print('Pos_cnt: %d, Neg_cnt: %d.'%(pos_cnt, neg_cnt))

data_train = np.array(data_train)
data_train = data_train[:,np.newaxis]
data_label = np.array(data_label)
data_label = data_label[:,np.newaxis]

np.save('traindata_%d.npy'%(I_VAL),data_train)
np.save('trainlabel_%d.npy'%(I_VAL),data_label)
