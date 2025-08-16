import math
import torch
import torch.nn as nn
from params import params_ad
import os

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_norm(num_channels, num_groups):
    return nn.GroupNorm(num_groups, num_channels)


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device      = x.device
        half_dim    = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb = None, rgb_condition_emb = None, csi_condition_emb = None):
        if x.shape[2] % 2 == 1:
            raise ValueError("downsampling tensor height should be even")
        if x.shape[3] % 2 == 1:
            raise ValueError("downsampling tensor width should be even")

        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
        
    def forward(self, x, time_emb = None):
        return self.upsample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, time_emb_dim, activation=SiLU(), num_groups=1, use_attention=False):
        super().__init__()

        self.activation = activation

        self.norm_1 = get_norm(in_channels, num_groups)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm_2 = get_norm(out_channels, num_groups)

        self.norm_3 = get_norm(out_channels, num_groups)
        self.conv_3 = nn.Sequential(
            nn.Dropout(p=dropout), 
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.time_bias  = nn.Linear(time_emb_dim, out_channels)

        self.residual_connection    = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x, time_emb):

        out = self.activation(self.norm_1(x))
        out = self.conv_1(out)

        out += self.time_bias(self.activation(time_emb))[:, :, None, None]
        out = self.activation(self.norm_2(out))
        out = self.conv_2(out)
        out = self.activation(self.norm_3(out))
        out = self.conv_3(out) + self.residual_connection(x)

        return out



class UNet(nn.Module):
    def __init__(
        self, img_channels = 2, base_channels=16, channel_mults=(1, 2, 4),
        num_res_blocks=3, time_emb_dim=128, time_emb_scale=1.0, activation=SiLU(),
        dropout=0., attention_resolutions=(1,), num_groups=4):
        super().__init__()

        self.activation = activation
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.init_conv  = nn.Conv2d(img_channels, base_channels, [3, 2], stride = [1], padding=[3,0])

        self.downs      = nn.ModuleList()
        self.ups        = nn.ModuleList()

        channels        = [base_channels]
        now_channels    = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels, out_channels, dropout,
                        time_emb_dim=time_emb_dim, activation=activation,
                        num_groups=num_groups, use_attention=i in attention_resolutions,
                    )
                )
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)


        self.mid = nn.ModuleList(
            [
                ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, activation=activation,
                    num_groups=num_groups, use_attention=True,
                ),
                ResidualBlock(
                    now_channels, now_channels, dropout,
                    time_emb_dim=time_emb_dim, activation=activation, 
                    num_groups=num_groups, use_attention=False,
                ),
            ]
        )

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels, out_channels, dropout, 
                    time_emb_dim=time_emb_dim, activation=activation, 
                    num_groups=num_groups, use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(Upsample(now_channels))

        assert len(channels) == 0

        self.out_conv1 = nn.Conv2d(base_channels, base_channels, 7, padding=[1, 3])
        self.up = Upsample(base_channels)
        self.out_conv2 = nn.Conv2d(base_channels, 1, [3, 2], padding=[1, 1])


    def forward(self, x, time, condition):

        time_emb = self.time_mlp(time)

        x = x.unsqueeze(1)
        condition = condition.unsqueeze(1)

        x = torch.cat([x, condition], dim=1)

        x = self.init_conv(x)


        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb)
            skips.append(x)
                
        for layer in self.mid:
            x = layer(x, time_emb)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb)

        x = self.out_conv1(x)
        x = self.activation(x)
        x = self.out_conv2(x)

        x = x.squeeze(1)

        return x


class AD_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder_0 = UNet()
        self.encoder_1 = UNet()
    
    def forward(self, data_t_0, data_1, data_t_1, data_0, t):

        predicted_0 = self.encoder_0(data_t_0, t, data_1)
        predicted_1 = self.encoder_1(data_t_1, t, data_0)

        return predicted_0, predicted_1



# if __name__ == "__main__":

#     os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    
#     device = "cuda:0"
#     # device = "cpu"
#     model = AD_model(params_ad).to(device)

#     para = sum(p.numel() for p in model.parameters())
#     print("Num params: ", para) 
#     print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

#     batch_size = 32

#     x = torch.zeros((batch_size, 100, 121), device=device).to(device)
#     t = torch.randint(0, params_ad.max_step, [batch_size], dtype=torch.int64, device = device).to(device)

#     output_0, output_1= model(x, x, x, x, t)

#     print(x.shape, output_0.shape, output_1.shape)

#     # pynvml.nvmlInit()
#     # handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
#     # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     # print(meminfo.used/1024**3, "G")  #已用显存大小

