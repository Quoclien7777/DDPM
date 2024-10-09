
from tqdm import tqdm
import numpy as np
import torch
import time
import os
import random
import matplotlib.pyplot as plt

from config import *
from embedding import *
from guassian_diffusion import *
from diffwave import *



def main():

    dataset = get_2d_dataset(CONFIG['data_path'],label_map)
    train_dataset = torch.utils.data.DataLoader(dataset = dataset,
                     batch_size = BATCH_SIZE,
                     shuffle = True)

    model =SimpleUnet().to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(),lr = LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    scheduler = CosineNoiseScheduler(TIMESTEPS)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    checkpoint = torch.load(os.path.join(CONFIG['case01'],'checkpoint_428.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    his = []
    for epoch in range(NUM_EPOCHS):
        losses = []
        for _,data in enumerate(tqdm(train_dataset)):
            labels = data['label']
            data = data['data'].float()
            optimizer.zero_grad()
            im = data.to(device)

            t = torch.randint(0,TIMESTEPS,(im.shape[0],))
            '''t = torch.normal(0.0,0.4,size=(im.shape[0],))

            t = (t - torch.min(t)) / (torch.max(t) - torch.min(t))
            t = torch.abs(t)
            t = t * 200
            t = t.int()'''
            noisy_im,noise = scheduler.add_noise(data,t)
            noise_pred = model(noisy_im.to(device),t.to(device)).to(device)

            loss =criterion(noise_pred ,noise.to(device))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        his.append(np.mean(losses))
        print('Finish epoch:{}, loss = {:.6f}'.format(epoch+1,np.mean(losses) ))

        if True:#(epoch + 1) % 5 ==0:
            torch.save(
                {
                    'model_state_dict' : model.state_dict(),


                },os.path.join(CONFIG['case01'],f'checkpoint_{epoch+429}.pth')
        )
            random_data = torch.normal(0.0,1.0,size =(1,data.shape[1],data.shape[2],data.shape[3]))
            syn_img = []

            for i in reversed(range(TIMESTEPS)):
                with torch.no_grad():

                    pred_noise = model(random_data.to(device), (i* torch.ones(im.shape[0])).to(device)).to('cpu')

                random_data = scheduler.predict_prev_steps(random_data,pred_noise,i)

                assert not torch.any(random_data.isnan())
                if i % 20 == 0:
                    print(i,'-',torch.mean(pred_noise).item(),'_',torch.var(pred_noise).item())
                    a = (random_data - torch.min(random_data))/(torch.max(random_data) - torch.min(random_data))
                    save_img = torch.zeros(3,640,640)

                    for x in range(3):
                        for y in range(64):
                            for z in range(64):
                                save_img[x,y*10:(y+1)*10,z*10:(z+1)*10] = a[0,x,y,z]
                    plt.imsave(os.path.join(CONFIG['save_img'], f'img_epoch_{epoch + 1}_img_timesteps_{i + 1}.png'),save_img.permute(1,2,0).detach().numpy())
                    plt.close()

            random_data = (random_data - torch.min(random_data) + random_data - torch.max(random_data)) /(torch.max(random_data) - torch.min(random_data))
            gasf,gadf,mtf =image_to_ecg(random_data[0,:,:,:])
            plt.plot(gasf,label ='gasf')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(CONFIG['save_ecg'],f'ecg_epoch_{epoch+1}.png'))
            plt.close()



if __name__ == '__main__':
    main()
