import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from config import *
from embedding import *
from guassian_diffusion import *

if __name__ == "__main__":
    gen_img = torch.normal(0.0,1.0,size=(1,3,64,64))
    model = SimpleUnet()
    checkpoint = torch.load(os.path.join(CONFIG['case00']),'checkpoint_100_uncondition.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    schedular = CosineNoiseScheduler(num_timesteps = TIMESTEPS)
    model.eval()
    
    for i in reversed(range(TIMESTEPS//12)):
        noise_pred = model(gen_img,i)
        gen_img = schedular.predict_prev_steps(gen_img,noise_pred,i)

    gen_img = gen_img[0,:,:,:].cpu().detach().numpy()
    gen_ecg = image_to_ecg(gen_img)
    plt.figure(figsize = (20,3),constrained_layout = True)
    plt.plot(gen_ecg)
    plt.save_fig(os.path.join(CONFIG['test_result'],'sample_0.png'))
    plt.close()


