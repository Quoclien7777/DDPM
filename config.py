import torch
import os
CONFIG = {
    'data_path' : '/usr/diem/Documents/data/mit',
    'wgan_path' : '/usr/diem/Documents/LHQuoc/ecg_ddpm/wgan_gen_data',
    'case00' : '/usr/diem/Documents/LHQuoc/ecg_ddpm/case00',
    'case01' : '/usr/diem/Documents/LHQuoc/ecg_ddpm/case01',
    'save_img' : '/usr/diem/Documents/LHQuoc/ecg_ddpm/save_img/',
    'save_ecg' : '/usr/diem/Documents/LHQuoc/ecg_ddpm/save_ecg/',
    '10s_csv'  : '/usr/diem/Document/LHQuoc/ecg_ddpm/10s/csv/10s_mitdb.csv',
    '10s_model': '/usr/diem/Documents/LHQuoc/ecg_ddpm/10s/ckpt',
    '10s_img'  : '/usr/diem/Documents/LHQuoc/ecg_ddpm/10s/img',
    '10s_ecg'  : '/usr/diem/Documents/LHQuoc/ecg_ddpm/10s/ecg',
}


label_map ={
     "N":0,"L" :1,
}

device = 'cuda'
torch.cuda.empty_cache()
NUM_EPOCHS = 300
BATCH_SIZE = 2
CHANNELS_IMG = 3
IMAGE_SIZE = 64
FEATURES_DISC = 64
FEATURES_GEN = 64
LEARNING_RATE = 1e-4
LAMBDA_GP = 10
Z_DIM = 100
ratio = 0.8
#ddpm
TIMESTEPS = 200
SAMPLING_RATE = 360

HR_METHODS = {
    '1RR': 1,
    '3RR': 3,
    '6RR': 6
}
