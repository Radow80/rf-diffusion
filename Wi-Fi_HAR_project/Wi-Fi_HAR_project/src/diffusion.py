import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from params import params_ad
from einops import rearrange


class GaussianDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.params = params
        self.input_dim = self.params.sample_rate # input time-series data length, N
        self.extra_dim = self.params.extra_dim # dimension of each data sample, e.g., [S A 2] for complex-valued CSI
        self.max_step = self.params.max_step # maximum diffusion steps
        beta = np.array(self.params.noise_schedule) # \beta, [T]
        self.beta = beta
        alpha = torch.tensor((1-beta).astype(np.float32)) # \alpha_t [T]
        self.alpha = alpha
        self.alpha_bar = torch.cumprod(alpha, dim=0) # \bar{\alpha_t}, [T]

        # The overall weight of gaussian noise \epsilon in degraded data x_t
        self.noise_weights = torch.sqrt(1 - self.alpha_bar) # \sqrt{1 - \bar{\alpha_t}}, [T]
        self.info_weights = torch.sqrt(self.alpha_bar) # \sqrt{\bar{\alpha_t}}, [T]

        # add params
        self.reciprocal_sqrt_alphas = torch.sqrt(1 / alpha)
        self.remove_noise_coeff = beta / torch.sqrt(1 - self.alpha_bar)

        # Imputation
        self.seglen = self.params.seglen
        
        self.loss_fn = nn.MSELoss()
        self.l1_loss_fn = nn.L1Loss()


    # def degrade_fn(self, data, t, noise):
    #     device = data.device
    #     noise_weight = self.noise_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    #     info_weight = self.info_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    #     x_t = info_weight * data + noise * noise_weight
    #     return x_t

    def degrade_fn(self, data, t, noise):
        device = data.device
        noise_weight = self.noise_weights[t].unsqueeze(-1).unsqueeze(-1).to(device)
        info_weight = self.info_weights[t].unsqueeze(-1).unsqueeze(-1).to(device)
        x_t = info_weight * data + noise * noise_weight
        return x_t


    # def split(self, data):
    #     data = rearrange(data, "B (S L) D I -> B S L D I", L = self.seglen * 2)
    #     data_0, data_1 = data[:, :, :self.seglen], data[:, :, self.seglen:]
    #     data_0 = rearrange(data_0, "B S L D I -> B (S L) D I")
    #     data_1 = rearrange(data_1, "B S L D I -> B (S L) D I")
    #     return data_0, data_1

    def split(self, data):
        data = rearrange(data, "B (S L) D -> B S L D", L = self.seglen * 2)
        data_0, data_1 = data[:, :, :self.seglen], data[:, :, self.seglen:]
        data_0 = rearrange(data_0, "B S L D -> B (S L) D")
        data_1 = rearrange(data_1, "B S L D -> B (S L) D")
        return data_0, data_1

    # def merge(self, data_0, data_1):
    #     data_0 = rearrange(data_0, "B (S L) D I -> B S L D I", L = self.seglen)
    #     data_1 = rearrange(data_1, "B (S L) D I -> B S L D I", L = self.seglen)
    #     data = torch.concat((data_0, data_1), dim = 2)
    #     data = rearrange(data, "B S L D I -> B (S L) D I")
    #     return data

    def merge(self, data_0, data_1):
        data_0 = rearrange(data_0, "B (S L) D -> B S L D", L = self.seglen)
        data_1 = rearrange(data_1, "B (S L) D -> B S L D", L = self.seglen)
        data = torch.concat((data_0, data_1), dim = 2)
        data = rearrange(data, "B S L D -> B (S L) D")
        return data


    def imputation_train(self, data, model):
        device = data.device
        B = data.shape[0]
        t = torch.randint(0, self.max_step, [B], dtype=torch.int64)
        data_0, data_1 = self.split(data)
        noise = torch.randn_like(data_0, dtype=torch.float32, device=device)

        data_t_0 = self.degrade_fn(data_0, t, noise)
        data_t_1 = self.degrade_fn(data_1, t, noise)
        predicted_0, predicted_1 = model(data_t_0, data_1, data_t_1, data_0, t)
        predicted = self.merge(predicted_0, predicted_1)
        loss = self.loss_fn(data, predicted)

        # data_abs_sq = torch.abs(data[...,0]) + torch.abs(data[...,1])
        # predicted_abs_sq = torch.abs(predicted[...,0]) + torch.abs(predicted[...,1])
        # loss_1 = self.loss_fn(data_abs_sq, predicted_abs_sq)

        return loss, t
    
    
    def data_sample(self, restore_fn, data, device):
        with torch.no_grad():
            batch_size = data.shape[0] # B
            data_0, data_1 = self.split(data)
            x_0, x_1 = data_0.to(device), data_1.to(device)
            with torch.no_grad():
                for s in range(self.max_step-1, -1, -1): # reverse from t to 0
                    noise = torch.randn_like(x_0, dtype=torch.float32, device=device)
                    x_0 = self.degrade_fn(x_0, t=s*torch.ones(batch_size, dtype=torch.int64), noise = noise)
                    x_1 = self.degrade_fn(x_1, t=s*torch.ones(batch_size, dtype=torch.int64), noise = noise) 
                    x_0, x_1 = restore_fn(x_0, data_1, x_1, data_0, s*torch.ones(batch_size, dtype=torch.int64, device=device))
            x = self.merge(x_0, x_1)
            return x, data
    

    def fast_sample(self, restore_fn, data, device):
        batch_size = data.shape[0] # B
        data_0, data_1 = self.split(data)
        x_0, x_1 = data_0.to(device), data_1.to(device)
        with torch.no_grad():
            s = self.max_step-1
            noise = torch.randn_like(x_0, dtype=torch.float32, device=device)
            x_0 = self.degrade_fn(x_0, t=s*torch.ones(batch_size, dtype=torch.int64), noise = noise)
            x_1 = self.degrade_fn(x_1, t=s*torch.ones(batch_size, dtype=torch.int64), noise = noise) 
            x_0, x_1 = restore_fn(x_0, data_1, x_1, data_0, s*torch.ones(batch_size, dtype=torch.int64))
        x = self.merge(x_0, x_1)
        return x, data
    

    def show_add_noise(self, data, device):
        batch_size = data.shape[0] # B
        res_list = [data]
        with torch.no_grad():
            for s in range(self.max_step-1):
                noise = torch.randn_like(data, dtype=torch.float32, device=device)
                data_temp = self.degrade_fn(data, t=s*torch.ones(batch_size, dtype=torch.int64), noise = noise)
                res_list.append(data_temp)
        return res_list


# sd = GaussianDiffusion(params_ad)
# # model = AD_model(params_ad)
# data = torch.ones(1, 10, 3)
# for i in range(10):
#     data[:, i] = data[:, i] * i
# print(data)
# data_0, data_1 = sd.split(data)
# print(data_0, data_1)
# data_2 = sd.merge(data_0, data_1)
# print(data_2)
# # sd.imputation_train(data, model)


# from dataset_test_unfall import AD_Dataset
# from matplotlib import pyplot as plt

# sd = GaussianDiffusion(params_ad)
# dataset = AD_Dataset()
# data = dataset.__getitem__(3)
# data = data.unsqueeze(0)
# print(data.shape)
# res_list = sd.show_add_noise(data, 'cpu')
# print(len(res_list))
# save_dir = "/srv/csj/Anomaly_detection/imputation_DFS/add_noise_show/"
# for i in range(len(res_list)):
#     print(res_list[i].shape)
#     temp = rearrange(res_list[i], "1 a b -> b a")
#     plt.axis('off')
#     plt.pcolormesh(temp, shading='gouraud')
#     plt.savefig(save_dir + str(i) + ".png", bbox_inches='tight', pad_inches=0)
#     plt.clf()


