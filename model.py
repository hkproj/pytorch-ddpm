import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *


class DiffusionModel(pl.LightningModule):
    def __init__(self, in_size, t_range, img_channels):
        super().__init__()
        # Create a linear schedule from beta_small to beta_large
        self.beta_small = 1e-4
        self.beta_large = 0.02

        # The number of time steps in the diffusion process
        self.t_range = t_range
        self.in_size = in_size # (W * H)
        self.img_channels = img_channels


        bilinear = False
        self.inc = DoubleConv(img_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_channels)
        self.sa1 = SAWrapper(256, 8)
        self.sa2 = SAWrapper(256, 4)
        self.sa3 = SAWrapper(128, 8)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """

        # From image chanels to 64 channels.
        # (N, C, H, W) --> (N, 64, H, W)
        x1 = self.inc(x)

        # (N, 64, H, W) --> (N, 128, H/2, W/2)
        x2_temp = self.down1(x1)
        # (N, 128, H/2, W/2) + (N, 128, H/2, W/2) = (N, 128, H/2, W/2)
        x2 = x2_temp + self.pos_encoding(t, x2_temp.shape[1], x2_temp.shape[2])
        
        # (N, 128, H/2, W/2) --> (N, 256, H/4, W/4)
        x3_temp = self.down2(x2)
        # (N, 256, H/4, W/4) + (N, 256, H/4, W/4) = (N, 256, H/4, W/4)
        x3 = x3_temp + self.pos_encoding(t, x3_temp.shape[1], x3_temp.shape[2])

        # (N, 256, H/4, W/4) --> (N, 256, H/4, W/4)
        x3 = self.sa1(x3) 

        # (N, 256, N/4, W/4) --> (N, 256, H/8, W/8)
        x4_temp = self.down3(x3)
        # (N, 256, H/8, W/8) + (N, 256, H/8, W/8) = (N, 256, H/8, W/8)
        x4 = x4_temp + self.pos_encoding(t, x4_temp.shape[1], x4_temp.shape[2])

        # (N, 256, H/8, W/8) --> (N, 256, H/8, W/8)
        x4 = self.sa2(x4)

        # Output shape: (N, 128, H/4, W/4)
        x_temp = self.up1(x4, x3)
        # (N, 128, H/4, W/4) + (N, 128, H/4, W/4) = (N, 128, H/4, W/4)
        x = x_temp + self.pos_encoding(t, x_temp.shape[1], x_temp.shape[2])

        # (N, 128, H/4, W/4) --> (N, 128, H/4, W/4)
        x = self.sa3(x)

        # Output shape: (N, 64, H/2, W/2)
        x_temp = self.up2(x, x2)
        # (N, 64, H/2, W/2) + (N, 64, H/2, W/2) = (N, 64, H/2, W/2)
        x = x_temp + self.pos_encoding(t, x_temp.shape[1], x_temp.shape[2])

        # Output shape: (N, 64, H, W)
        x_temp = self.up3(x, x1)
        # (N, 64, H, W) + (N, 64, H, W) = (N, 64, H, W)
        x = x_temp + self.pos_encoding(t, x_temp.shape[1], x_temp.shape[2])
        
        # From 64 channels to image channels. 
        # (N, 64, H, W) --> (N, C, H, W)
        output = self.outc(x)

        return output

    def beta(self, t):
        return self.beta_small + (t / self.t_range) * (
            self.beta_large - self.beta_small
        )

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return math.prod([self.alpha(j) for j in range(t)])

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device) # Get random time steps, one for each sample in the batch. Shape: (N,)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device) # Random noise for each sample in the batch. Shape: (N, C, H, W)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i]) # Get alpha_bar for the time step "t" defined by ts[i]
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i]) # Generate the noisy image at time step "t" defined by ts[i]
            )
        noise_imgs = torch.stack(noise_imgs, dim=0) # Shape: (N, C, H, W)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float)) # Output shape: (N, C, H, W)
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size * self.img_channels), epsilons.reshape(-1, self.in_size * self.img_channels)
        )
        return loss

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape)
            else:
                z = 0
            e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1)) # Predict the noise for the current time step. Output shape: (N, C, H, W)
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
