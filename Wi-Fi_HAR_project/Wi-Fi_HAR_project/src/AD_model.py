import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pynvml
# from diffusion import GaussianDiffusion
from params import params_ad
from einops import rearrange
from complex.complex_module import *

def init_weight_norm(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def init_weight_zero(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def init_weight_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

@torch.jit.script
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_step, embed_dim), persistent=False)
        self.projection = nn.Sequential(
            ComplexLinear(embed_dim, hidden_dim, bias=True),
            ComplexSiLU(),
            ComplexLinear(hidden_dim, hidden_dim, bias=True),
        )
        self.hidden_dim = hidden_dim
        self.apply(init_weight_norm)
    def forward(self, t):
        x = self.embedding[t]
        return self.projection(x)
    def _build_embedding(self, max_step, embed_dim):
        steps = torch.arange(max_step).unsqueeze(1)  # [T, 1]
        dims = torch.arange(embed_dim).unsqueeze(0)  # [1, E]
        table = steps * torch.exp(-math.log(max_step) * dims / embed_dim)  # [T, E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table


# class PositionEmbedding(nn.Module):
#     def __init__(self, max_len, input_dim, hidden_dim):
#         super().__init__()
#         self.register_buffer('embedding', self._build_embedding(max_len, hidden_dim), persistent=False)
#         self.projection = ComplexLinear(input_dim, hidden_dim)
#         self.apply(init_weight_xavier)

#     def forward(self, x): 
#         x = self.projection(x)
#         y = self.embedding.to(x.device)
#         return complex_mul(x, y)

#     def _build_embedding(self, max_len, hidden_dim):
#         steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
#         dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
#         table = steps * torch.exp(-math.log(max_len) * dims / hidden_dim)     # [P,E]
#         table = torch.view_as_real(torch.exp(1j * table))
#         return table


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, input_dim, hidden_dim):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_len, hidden_dim), persistent=False)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.apply(init_weight_xavier)
    def forward(self, x): 
        x = self.projection(x)
        y = self.embedding.to(x.device)[..., 0]
        return torch.mul(x, y)
    def _build_embedding(self, max_len, hidden_dim):
        steps = torch.arange(max_len).unsqueeze(1)  # [P,1]
        dims = torch.arange(hidden_dim).unsqueeze(0)          # [1,E]
        table = steps * torch.exp(-math.log(max_len) * dims / hidden_dim)     # [P,E]
        table = torch.view_as_real(torch.exp(1j * table))
        return table




# class TrainablePositionEncoding(nn.Module):
#     def __init__(self, max_len, input_dim, hidden_dim):
#         super().__init__()
#         self.pe = nn.Parameter(torch.zeros(1, max_len, input_dim, 2))
#         self.linear = ComplexLinear(input_dim, hidden_dim)

#     def forward(self, x):
#         b, l, d, _ = x.shape
#         x = x + self.pe[:, :l]
#         x = self.linear(x)
#         return x


class DiA(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.hd = hidden_dim
        self.norm1 = ComplexLayerNorm(hidden_dim, eps=1e-9)
        self.s_attn = ComplexMultiHeadAttention(hidden_dim, hidden_dim, num_heads, dropout = 0.)
        self.norm2 = ComplexLayerNorm(hidden_dim, eps=1e-9)
        self.normc = ComplexLayerNorm(hidden_dim, eps=1e-9)
        self.x_attn = ComplexMultiHeadAttention(hidden_dim, hidden_dim, num_heads,  dropout = 0.)
        self.norm3 = ComplexLayerNorm(hidden_dim, eps=1e-9)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            ComplexLinear(hidden_dim, mlp_hidden_dim, bias=True),
            ComplexSiLU(),
            ComplexLinear(mlp_hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            ComplexSiLU(),
            ComplexLinear(hidden_dim, 6*hidden_dim, bias=True)
        )
        self.apply(init_weight_xavier)
        self.adaLN_modulation.apply(init_weight_zero)

        self.c_MLP = ComplexMLP(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x, t, c):
        """
        Embedding diffusion step t with adaptive layer-norm.
        Embedding condition c with cross-attention.
        - Input:\\
          x, [B, N, H, 2], \\ 
          t, [B, H, 2], \\
          c, [B, N, H, 2], \\
        """

        c = self.c_MLP(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        mod_x = modulate(self.norm1(x), shift_msa, scale_msa)

        x = x + gate_msa.unsqueeze(1) * self.s_attn(mod_x, mod_x, mod_x)
        x = x + self.x_attn(queries =self.normc(c), keys=self.norm2(x), values=self.norm2(x))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x



# class FinalLayer(nn.Module):
#     def __init__(self, hidden_dim, out_dim):
#         super().__init__()
#         self.norm = NaiveComplexLayerNorm(hidden_dim, eps=1e-9)
#         self.adaLN_modulation = nn.Sequential(
#             ComplexSiLU(),
#             ComplexLinear(hidden_dim, 2*hidden_dim, bias=True)
#         )
#         self.linear = ComplexLinear(hidden_dim, out_dim, bias=True)
#         self.apply(init_weight_zero)

#     def forward(self, x, t):
#         shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
#         x = x + modulate(self.norm(x), shift, scale)
#         x = self.linear(x)
#         return x



class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-9)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2*hidden_dim, bias=True)
        )
        self.linear = nn.Linear(hidden_dim, out_dim, bias=True)
        self.apply(init_weight_zero)

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = x + modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class MLP_real(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_dim = params.input_dim
        self.output_dim = self.input_dim
        self.hidden_dim = params.hidden_dim
        self.num_heads = params.num_heads
        self.dropout = params.dropout
        self.mlp_ratio = params.mlp_ratio
        self.p_embed = PositionEmbedding(params.sample_rate, params.input_dim, params.hidden_dim)
        self.t_embed = DiffusionEmbedding(params.max_step, params.embed_dim, params.hidden_dim)
        self.c_embed = PositionEmbedding(params.sample_rate, params.cond_dim, params.hidden_dim)
        self.blocks = nn.ModuleList([
            DiA(self.hidden_dim, self.num_heads, self.mlp_ratio) for _ in range(params.num_block)
        ])
        self.final_layer = FinalLayer(self.hidden_dim, self.output_dim)

        self.init_layer_x = MLP_real(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.init_layer_c = MLP_real(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.c2r = Complex2Real()
        self.c2r_t = Complex2Real()

        self.r2c_x = Real2Complex()
        self.r2c_c = Real2Complex()

        self.init_layer_c.apply(init_weight_xavier)
        self.init_layer_x.apply(init_weight_xavier)
        self.c2r.apply(init_weight_xavier)
        self.c2r_t.apply(init_weight_xavier)
        self.r2c_x.apply(init_weight_xavier)
        self.r2c_c.apply(init_weight_xavier)



    def forward(self, x, t, c):
        x = self.init_layer_x(self.p_embed(x))
        t = self.t_embed(t)
        c = self.init_layer_c(self.c_embed(c))
        x = self.r2c_x(x)
        c = self.r2c_c(c)
        for block in self.blocks:
            x = block(x, t, c)
        x = self.c2r(x)
        x = self.final_layer(x, self.c2r_t(t))
        return x
    


class AD_model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.encoder_0 = Encoder(params)
        self.encoder_1 = Encoder(params)
    
    def forward(self, data_t_0, data_1, data_t_1, data_0, t):

        # z_tensor = torch.zeros(data_0.shape).to(data_0.device)
        # data_0 = torch.stack((data_0, z_tensor), dim=-1)
        # data_1 = torch.stack((data_1, z_tensor), dim=-1)
        # data_t_0 = torch.stack((data_t_0, z_tensor), dim=-1)
        # data_t_1 = torch.stack((data_t_1, z_tensor), dim=-1)

        predicted_0 = self.encoder_0(data_t_0, t, data_1)
        predicted_1 = self.encoder_1(data_t_1, t, data_0)

        return predicted_0, predicted_1

    # def forward(self, data_t_0, data_1, data_t_1, data_0, t):

    #     predicted_0 = self.encoder_0(data_t_0, t, data_1)
    #     predicted_1 = self.encoder_1(data_t_1, t, data_0)

    #     return predicted_0, predicted_1



if __name__ == "__main__":


    import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"


    device = "cuda:0"
    # device = "cpu"
    model = AD_model(params_ad).to(device)
    para = sum(p.numel() for p in model.parameters())
    print("Num params: ", para) 
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

    batch_size = 1

    x = torch.zeros((batch_size, 100, 121), device=device).to(device)
    t = torch.randint(0, params_ad.max_step, [batch_size], dtype=torch.int64, device = device).to(device)

    output_0, output_1= model(x, x, x, x, t)

    # print(output_0.shape, output_1.shape)

    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
    # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # print(meminfo.used/1024**3, "G")  #已用显存大小


    import thop
    macs, params = thop.profile(model, inputs=(x, x, x, x, t,))
    gflops = macs / 1e9

    print(f"Model GFLOPs: {gflops:.2f} GFLOPs")
