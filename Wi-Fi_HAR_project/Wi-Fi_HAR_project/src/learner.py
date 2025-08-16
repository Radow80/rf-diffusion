import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from diffusion import GaussianDiffusion
from params import params_ad
from dataset_train import from_path
import dataset_test_unfall
import dataset_test_fall
from skimage.metrics import structural_similarity as ssim

if params_ad.is_complex:
    from AD_model import AD_model
else:
    from unet import AD_model


class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.tb_dir = params_ad.tb_dir
        self.log_dir = log_dir
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.device = model.device
        self.diffusion = GaussianDiffusion(params)

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, params.lr_step_size, gamma=params.lr_gamma)
        self.params = params
        self.iter = 0
        self.is_master = True
        self.loss_fn = nn.MSELoss()
        self.summary_writer = None


    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.iter = state_dict['iter']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_iter=None):

        self.restore_from_checkpoint(filename='continue_from_here')#training will continue from here
        self.model.train()
        while True:  # epoch
            # for data in tqdm(self.dataset, desc=f'Epoch {self.iter // len(self.dataset)}') if self.is_master else self.dataset:
            for data in self.dataset:
                if max_iter is not None and self.iter >= max_iter:
                    return
                      
                loss, t = self.train_iter(data)
                
                if self.is_master:
                    if self.iter % 1 == 0:
                        self._write_summary(self.iter, loss)
                    if self.iter % 1 == 0:
                        self._write_log(self.iter, loss, t)
                    if self.iter % (len(self.dataset) * 10) == 0:
                        self.save_to_checkpoint()
                        print(self.iter)
                        print("loss: ", loss)
                    #     with torch.no_grad():
                    #         self.test_unfall(self.iter, self.device)
                    #         self.test_fall(self.iter, self.device)

                self.iter += 1
                
            self.lr_scheduler.step()

    def train_iter(self, data):
        self.optimizer.zero_grad()

        data = data.to(self.model.device)

        #######################################################################################
        loss, t = self.diffusion.imputation_train(data, self.model)
        #######################################################################################

        if torch.isnan(loss).any():
            raise RuntimeError(f'Detected NaN loss at iteration {self.iter}.')
        
        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.optimizer.step()
        return loss, t

    def _write_log(self, iter, loss, t):
        log_path = os.path.join(params_ad.log_dir, 'pleasework' + '.txt')
        with open(log_path, 'a', encoding='utf-8') as file:
            file.write(f'train/loss,    {loss},     {iter}\n')


    def _write_summary(self, iter, loss):
        writer = self.summary_writer or SummaryWriter(self.tb_dir, purge_step=iter)
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/grad_norm', self.grad_norm, iter)
        writer.flush()
        self.summary_writer = writer


    def test_fall(self, iter, device):
        loss_fn = nn.MSELoss()
        device = device
        model = AD_model(params_ad)
        model.load_state_dict(torch.load(params_ad.model_dir + "weights.pt")['model'])
        model = model.to(device)
        model.eval()
        diffusion = GaussianDiffusion(params_ad)
        dataset = dataset_test_fall.from_path_eval(params_ad)
        i = 0
        ssim_sum = 0
        for data in tqdm(dataset): 
            if i == 10:
                break
            data = data.to(device)
            predicted, data = diffusion.data_sample(model, data, device)

            loss = loss_fn(data, predicted)
            # data = torch.sqrt(data[...,0]**2 + data[...,1]**2)
            # predicted = torch.sqrt(predicted[...,0]**2 + predicted[...,1]**2)
            ssim_value = ssim(data[0].cpu().numpy(), predicted[0].cpu().numpy(), data_range=1)
            ssim_sum += ssim_value
            i += 1
        ssim_sum /= 10
        writer = self.summary_writer or SummaryWriter(self.tb_dir, purge_step=iter)
        writer.add_scalar('test_fall/ssim', ssim_sum, iter)
        writer.flush()
        self.summary_writer = writer


    def test_unfall(self, iter, device):
        loss_fn = nn.MSELoss()
        device = device
        model = AD_model(params_ad)
        model.load_state_dict(torch.load(params_ad.model_dir + "weights.pt")['model'])
        model = model.to(device)
        model.eval()
        diffusion = GaussianDiffusion(params_ad)
        dataset = dataset_test_unfall.from_path_eval(params_ad)
        i = 0
        ssim_sum = 0
        for data in tqdm(dataset): 
            if i == 10:
                break
            data = data.to(device)
            predicted, data = diffusion.data_sample(model, data, device)

            loss = loss_fn(data, predicted)
            # data = torch.sqrt(data[...,0]**2 + data[...,1]**2)
            # predicted = torch.sqrt(predicted[...,0]**2 + predicted[...,1]**2)
            ssim_value = ssim(data[0].cpu().numpy(), predicted[0].cpu().numpy(), data_range=1)
            ssim_sum += ssim_value
            i += 1
        ssim_sum /= 10
        writer = self.summary_writer or SummaryWriter(self.tb_dir, purge_step=iter)
        writer.add_scalar('test_unfall/ssim', ssim_sum, iter)
        writer.flush()
        self.summary_writer = writer