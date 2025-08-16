# -*- coding: utf-8 -*-
import os
import torch
from params import params_ad
if params_ad.is_complex:
    from AD_model import AD_model
else:
    from unet import AD_model

# from imputation_vae import AD_model_VAE as AD_model
# from imputation_vae import ImputationVAE as GaussianDiffusion
from diffusion import GaussianDiffusion
from tqdm import tqdm
import torch.nn as nn
from dataset_test_fall import from_path_eval
# from dataset_train import from_path_eval
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sewar.full_ref import uqi
from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import lpips
import torchvision.transforms as transforms

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch_home = os.path.join(base_dir, 'temp_torch_home')
os.environ['TORCH_HOME'] = torch_home


lpips_vgg = lpips.LPIPS(net='vgg')
lpips_alex = lpips.LPIPS(net='alex')
lpips_squeeze = lpips.LPIPS(net='squeeze')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    ])


def calculate_lpips_vgg(image1, image2):
    image1 = transform(image1)
    image2 = transform(image2)
    lpips_score = lpips_vgg(image1, image2)
    return lpips_score

def calculate_lpips_alex(image1, image2):
    image1 = transform(image1)
    image2 = transform(image2)
    lpips_score = lpips_alex(image1, image2)
    return lpips_score

def calculate_lpips_squeeze(image1, image2):
    image1 = transform(image1)
    image2 = transform(image2)
    lpips_score = lpips_squeeze(image1, image2)
    return lpips_score


test_type = "test_fall_" + str(params_ad.seglen) + "__" + str(params_ad.max_step) + "__" + str(params_ad.num_block) + "__" + str(params_ad.num_heads)

# test_type = "train_test_2_" + str(params_ad.seglen)

def calculate_cosine_similarity(image1, image2):
    vector1 = image1.flatten()
    vector2 = image2.flatten()
    cosine_similarity = np.dot(vector1, vector2)
    return cosine_similarity


def test(params):

    import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # print("NOTICE: ", params_ad.model_dir + "weights.pt")


    loss_fn = nn.MSELoss()
    device = torch.device("cuda:0")
    model = AD_model(params_ad)
    model_path = os.path.join(params.model_dir, 'weights.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])#changed map location, not sure if it helps, original was 'cpu'
    model = model.to(device)
    model.eval()
    diffusion = GaussianDiffusion(params)
    diffusion = diffusion.to(device)
    dataset = from_path_eval(params_ad)
    i = 0

    with torch.no_grad():
        for data in tqdm(dataset): 
            data = data.to(device)

            print("NOTICE0: ", data.shape)

            predicted, data = diffusion.data_sample(model, data, device)
            print(predicted.shape, data.shape)

            loss = loss_fn(data.cpu(), predicted.cpu())

            lpips_vgg_value = calculate_lpips_vgg(data.cpu(), predicted.cpu())
            lpips_alex_value = calculate_lpips_alex(data.cpu(), predicted.cpu())
            lpips_squeeze_value = calculate_lpips_squeeze(data.cpu(), predicted.cpu())


            data = data.cpu().numpy()
            predicted = predicted.cpu().numpy()

            ssim_value_3 = ssim(data, predicted, data_range=2, win_size=3, channel_axis=0)
            ssim_value_5 = ssim(data, predicted, data_range=2, win_size=5, channel_axis=0)
            ssim_value_7 = ssim(data, predicted, data_range=2, win_size=7, channel_axis=0)
            ssim_value_9 = ssim(data, predicted, data_range=2, win_size=9, channel_axis=0)
            ssim_value_11 = ssim(data, predicted, data_range=2, win_size=11, channel_axis=0)
            ssim_value_13 = ssim(data, predicted, data_range=2, win_size=13, channel_axis=0)
            ssim_value_15 = ssim(data, predicted, data_range=2, win_size=15, channel_axis=0)
            ssim_value_17 = ssim(data, predicted, data_range=2, win_size=17, channel_axis=0)
            ssim_value_19 = ssim(data, predicted, data_range=2, win_size=19, channel_axis=0)
            ssim_value_21 = ssim(data, predicted, data_range=2, win_size=21, channel_axis=0)
            uqi_value_3 = uqi(data[0], predicted[0], ws=3)
            uqi_value_5 = uqi(data[0], predicted[0], ws=5)
            uqi_value_7 = uqi(data[0], predicted[0], ws=7)
            uqi_value_9 = uqi(data[0], predicted[0], ws=9)
            uqi_value_11 = uqi(data[0], predicted[0], ws=11)
            uqi_value_13 = uqi(data[0], predicted[0], ws=13)
            uqi_value_15 = uqi(data[0], predicted[0], ws=15)
            uqi_value_17 = uqi(data[0], predicted[0], ws=17)
            uqi_value_19 = uqi(data[0], predicted[0], ws=19)
            uqi_value_21 = uqi(data[0], predicted[0], ws=21)
            psnr_value = psnr(data[0], predicted[0], data_range=2)
            cos_value = calculate_cosine_similarity(data[0], predicted[0])
            
            print(loss, ssim_value_21, uqi_value_21, psnr_value)
            _write_log(i, loss, ssim_value_3, ssim_value_5, ssim_value_7, ssim_value_9, ssim_value_11, ssim_value_13, ssim_value_15, ssim_value_17, ssim_value_19, ssim_value_21, uqi_value_3, uqi_value_5, uqi_value_7, uqi_value_9, uqi_value_11, uqi_value_13, uqi_value_15, uqi_value_17, uqi_value_19, uqi_value_21, psnr_value, cos_value, lpips_vgg_value, lpips_alex_value, lpips_squeeze_value, test_type)
            draw(data, i, test_type, "gt")
            draw(predicted, i, test_type, "pd")
            i += 1

def _write_log(i, loss, ssim_value_3, ssim_value_5, ssim_value_7, ssim_value_9, ssim_value_11, ssim_value_13, ssim_value_15, ssim_value_17, ssim_value_19, ssim_value_21, uqi_value_3, uqi_value_5, uqi_value_7, uqi_value_9, uqi_value_11, uqi_value_13, uqi_value_15, uqi_value_17, uqi_value_19, uqi_value_21, psnr_value, cos_value, lpips_vgg_value, lpips_alex_value, lpips_squeeze_value, test_type):
    log_path = os.path.join(params_ad.log_dir, test_type + '.txt')
    with open(log_path, 'a', encoding='utf-8') as file:
        line = (
            "test/loss, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n" #changed to 25 indices instead of 27
            .format(
                loss, ssim_value_3, ssim_value_5, ssim_value_7, ssim_value_9, ssim_value_11, ssim_value_13, ssim_value_15, ssim_value_17, ssim_value_19, ssim_value_21,
                uqi_value_3, uqi_value_5, uqi_value_7, uqi_value_9, uqi_value_11, uqi_value_13, uqi_value_15, uqi_value_17, uqi_value_19, uqi_value_21,
                psnr_value, cos_value, lpips_vgg_value, lpips_alex_value, lpips_squeeze_value, i
            )
        )
        file.write(line)

def draw(predicted, i, dir, type):
    save_dir = os.path.join(base_dir, 'pngs', dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    predicted = rearrange(predicted, "1 a b-> b a")
    plt.pcolormesh(predicted, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(os.path.join(save_dir, "{}_{}.png".format(i, type)))
    plt.clf()


if __name__ == '__main__':
    params = params_ad
    test(params)
