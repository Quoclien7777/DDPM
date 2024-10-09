import torch
from torch import nn
import math
import torch.nn.functional as F
class CosineNoiseScheduler():
  def __init__(self,num_timesteps,s=0.008,device='cpu'):
      self.timesteps = num_timesteps
      pi = math.pi
      
      self.clip_max =0.999
      self.clip_min = 1. - self.clip_max
      timesteps_cosine = (torch.arange(self.timesteps)) / self.timesteps +s
      alphas = timesteps_cosine / (1 + s) * pi /2
      alphas = torch.cos(alphas).pow(2)
      alphas = alphas / alphas[0]
      self.alphas =alphas
      alphas = torch.cat([torch.Tensor([1.0]),self.alphas[:self.timesteps -1]])

      self.betas = torch.clamp(1. - self.alphas/alphas,max= self.clip_max)
      self.alphas_cumprod = torch.clamp(torch.cumprod(alphas,axis =0),min = self.clip_min,max =self.clip_max)
      self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
      self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
      self.sqrt_alphas_cumprod  = torch.sqrt(self.alphas_cumprod)
      self.sqrt_one_minus_alphas_cumprod  = torch.sqrt(1. - self.alphas_cumprod)
      self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
      
  def add_noise(self,image,t):
      noise = torch.randn_like(image)
      image_shape = image.shape
      batch_size = image.shape[0]

      sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].reshape(batch_size)
      sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].reshape(batch_size)

      for _ in range(len(image_shape) - 1 ):
          sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
          sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
      result =sqrt_alpha_cumprod * image + sqrt_one_minus * noise

      return result,noise

  def predict_prev_steps(self,xt,noise_pred,t): # <200
      x_recon = torch.sqrt(1.0/self.alphas_cumprod[t]) * xt - torch.sqrt(1.0/self.alphas_cumprod[t] - 1) * noise_pred
      coef1 = self.betas[t] * torch.sqrt(self.alphas_cumprod_prev[t]) / (1.0 -self.alphas_cumprod[t])
      coef2 = (1. - self.alphas_cumprod_prev[t]) * torch.sqrt(self.alphas[t]) / (1.0 - self.alphas_cumprod[t])
      mean = coef1 * x_recon + coef2 * xt
      #mean = torch.sqrt(1.0/self.alphas[t]) * (xt - (self.betas[t]/self.sqrt_one_minus_alphas_cumprod[t]) * noise_pred) #
      posterior_variance =self.posterior_variance[t]
      if t==0:
          return mean

      else:
          noise =  torch.randn_like(xt)
          result = (mean + torch.sqrt(posterior_variance) *noise)
          return result



class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=((3 - 1) * 20) // 2,
                      dilation=20)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=((3 - 1) * 40) // 2,
                      dilation=40)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=((3 - 1) * 80) // 2,
                      dilation=80)
        )
        self.transform = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x_cat = torch.cat([x1, x2, x3, x4], 1)
        x_res = self.transform(x_cat)
        return self.relu(x + x_res)


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.up = up
        if up:
            self.time_mlp = nn.Linear(time_emb_dim, 2 * in_channels)
            self.block = InceptionResidualBlock(2 * in_channels)
            self.transform = nn.ConvTranspose2d(2 * in_channels, out_channels, 4, 2, 1)
        else:
            self.time_mlp = nn.Linear(time_emb_dim, in_channels)
            self.block = InceptionResidualBlock(in_channels)
            self.transform = nn.Conv2d(in_channels, out_channels, 4, 2, 1)

        self.relu = nn.ReLU()

    def forward(self, x, t):
        x = self.block(x)

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 1]
        time_emb = time_emb.reshape(time_emb.shape[0],time_emb.shape[1],time_emb.shape[2],1)
        
        x = x + time_emb

        x = self.transform(x)

        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, n_channels):
        super(PositionalEncoding, self).__init__()
        self.n_channels = n_channels
        self.LINEAR_SCALE = 5000

    def forward(self, noise_level):
        if len(noise_level.shape) > 1:
            noise_level = noise_level.squeeze(-1)
        half_dim = self.n_channels // 2
        exponents = torch.arange(half_dim, dtype=torch.float32).to(noise_level) / float(half_dim)
        exponents = 1e-4 ** exponents
        exponents = self.LINEAR_SCALE * noise_level.unsqueeze(1) * exponents.unsqueeze(0)
        return torch.cat([exponents.sin(), exponents.cos()], dim=-1)


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (24, 48, 96, 192)
        up_channels = ( 192, 96, 48, 24)
        out_dim = 1
        time_emb_dim = 64
        # time_emb_dim = 128

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            # nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # self.time_mlp = nn.Sequential(
        #     PositionalEncoding(time_emb_dim),
        #     nn.Linear(time_emb_dim, time_emb_dim),
        #     nn.Linear(time_emb_dim, time_emb_dim),
        #     nn.ReLU()
        # )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([UnetBlock(down_channels[i], down_channels[i + 1], time_emb_dim) \
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([UnetBlock(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) \
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:           
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)

        return self.output(x)


if __name__ == '__main__':
    model = SimpleUnet()
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    x = torch.rand((50, 3,64, 64))
    out = model(x,torch.Tensor(50))
    print(out.shape)