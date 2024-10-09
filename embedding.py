import math
from config import SAMPLING_RATE
import numpy as np
from pyts.image import MarkovTransitionField
import wfdb
import os
import matplotlib.pyplot as plt
import time
class EcgDataset():
    def __init__(self,length,dataset,label):
        self.length = length 
        self.dataset = dataset
        self.label = label
        
    def __len__(self):
        return self.length
        
    def __getitem__(self,idx):
        return {'data' : self.dataset[idx],
               'label' : self.label[idx]}
        

def get_10s_2d_dataset(link):
    data = []
    
    list_file = os.listdir(link)
    for i in list_file:
        if '.dat' in i:
            print(i[:3])
            record = wfdb.rdrecord(link + f'/{i[0:3]}')
            ann = wfdb.rdann(link+ f'/{i[:3]}', extension='atr')
            ann = ann.__dict__
            samp = ann['sample']
            sym = ann['symbol']
            channel =0
            if i == '114' or i == '102':
                channel =1
            current_mark = samp[0]
            data_segment = record.p_signal[samp[0]:(samp[0] + SAMPLING_RATE * 10),channel]

            while current_mark < record.sig_len :
                if data_segment.shape[0] == SAMPLING_RATE * 10 :
                    f = []
                    for i in range(len(data_segment)):
                        if i % 3 ==0:
                            f.append(data_segment[i])
                    data.append(ecg_to_image(f))
                
                current_mark += SAMPLING_RATE * 10
                
    return data

def get_2d_dataset(link,label_map):
    data = []
    labels = []
    list_file = os.listdir(link)
    for i in list_file:
        if '.dat' in i:
            record = wfdb.rdrecord(link + f'/{i[0:3]}')
            ann = wfdb.rdann(link+ f'/{i[:3]}', extension='atr')
            ann = ann.__dict__
            samp = ann['sample']
            sym = ann['symbol']
            channel =0
            if i == '114':
                channel =1
            data_segment=  record.p_signal[:,channel]
            for count in range(len(sym)):
                if data_segment[(samp[count] - 32):(samp[count] + 32)].shape[0] == 64 and sym[count] in label_map.keys():
                     data.append(ecg_to_image(data_segment[(samp[count] - 32):(samp[count] + 32)]))

                     labels.append(label_map[sym[count]])
                
        
    dataset = EcgDataset(len(labels),data,labels)
    return dataset

def get_1d_dataset(link,label_map):
    list_file = os.listdir(link)
    label=[]
    dataset=[]
    for i in list_file:
        if '.dat' in i:
            channel = 0
            if i == '114':
                channel = 1
            data = wfdb.rdrecord(link + f'/{i[0:3]}').p_signal[:,channel]
            ann = wfdb.rdann(link + f'/{i[:3]}', extension='atr')
            ann = ann.__dict__
            samp = ann['sample']
            sym = ann['symbol']
            for count in range(len(sym)):
                mark = samp[count]
                if data[(mark - 32):(mark + 32)].shape[0] == 64 and sym[count] in label_map.keys():
                    dataset.append(data[(mark - 32):(mark + 32)])
                    label.append(label_map[sym[count]])

    dataset = np.array(dataset)
    label = np.array(label)
    return dataset, label

def ecg_to_image(data):
    # return into 3 channel 2d data
    length = len(data)# len(data.p_signal)
    PI = math.pi
    min = np.min(data)
    max = np.max(data)
    if max == min:
        norm_data = np.zeros((3,length,length))
        return  norm_data
    else:
        norm_data = (data - min + data - max) / (max - min)
        norm_data = np.clip(norm_data,-0.999,0.999)
    phi_rad = np.arccos(norm_data) 

    gasf = np.array(np.cos(0.5 *(phi_rad[:, np.newaxis] + phi_rad))  , dtype=np.float32).reshape(1,length,length)
    gadf = np.array(np.cos(0.5 * (phi_rad[:, np.newaxis] - phi_rad))  , dtype=np.float32).reshape(1,length,length)
    mtf = MarkovTransitionField(n_bins=4)
    X_mtf = mtf.fit_transform(np.array([data], dtype=np.float32)).reshape(1,length,length)
    return np.concatenate((gasf, gadf, X_mtf), axis=0, dtype=np.float32)


def image_to_ecg(synthetic_image):
    gasf = np.diagonal(synthetic_image[ 0, :, :])
    gadf = np.diagonal(synthetic_image[ 1, :, :])
    mtf = np.diagonal(synthetic_image[2,:,:])
    
    synthetic = np.arccos(gasf)
    return np.max(synthetic) - synthetic,gadf,mtf
