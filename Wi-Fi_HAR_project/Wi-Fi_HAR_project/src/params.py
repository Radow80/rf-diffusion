import numpy as np
import os

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

# 基础配置
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')
model_dir = os.path.join(base_dir, 'models')
log_dir = os.path.join(base_dir, 'logs')

# 确保必要的目录存在
for dir_path in [data_dir, model_dir, log_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

params_ad = AttrDict(
    is_complex = True,
    gpus_to_use = [0],  # 默认使用第一个GPU

    # 路径配置
    log_dir = os.path.join(log_dir, 'train.log'),
    tb_dir = os.path.join(log_dir, 'tensorboard'),
    model_dir = os.path.join(model_dir, 'weights'),

    # 数据路径
    unfall_train_txt = os.path.join(data_dir, 'train/train.txt'),
    unfall_test_txt = os.path.join(data_dir, 'test_unfall/test.txt'),
    fall_test_txt = os.path.join(data_dir, 'test_fall/test.txt'),

    train_fast_path = os.path.join(data_dir, 'train/fast/'),
    test_unfall_fast_path = os.path.join(data_dir, 'test_unfall/fast/'),
    test_fall_fast_path = os.path.join(data_dir, 'test_fall/fast/'),

    # 训练参数
    max_iter = 50000,
    batch_size = 32,
    learning_rate = 2e-4,
    max_grad_norm = None,

    # 数据参数
    sample_rate = 100,  # 200//2
    input_dim = 121,
    extra_dim = [121],
    cond_dim = 121,

    # 模型参数
    embed_dim = 512,
    hidden_dim = 512,
    num_heads = 16,
    num_block = 8,
    dropout = 0.,
    mlp_ratio = 4,

    # 时间序列参数
    seglen = 20,

    # 扩散模型参数
    max_step = 50,
    noise_schedule = np.linspace(1e-4 * 1000 / 50, 0.02 * 1000 / 50, 50).tolist(),

    # 学习率调度
    lr_step_size = 14,
    lr_gamma = 0.5,
) 