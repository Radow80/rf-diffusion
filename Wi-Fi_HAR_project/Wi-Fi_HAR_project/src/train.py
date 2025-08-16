import os

import torch
# from torch.cuda import device_count
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel

from params import params_ad
from learner import tfdiffLearner

from dataset_train import from_path

if params_ad.is_complex:
    from AD_model import AD_model
else:
    from unet import AD_model

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

def _train_impl(replica_id, model, dataset, params):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
    learner = tfdiffLearner(params.log_dir, params.model_dir, model, dataset, opt, params)
    learner.is_master = (replica_id == 0)
    learner.train(max_iter=params.max_iter)

def train_distributed(replica_id, replica_count, port, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
    dataset = from_path(params, is_distributed=True)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = AD_model(params_ad).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id], find_unused_parameters=True)
    _train_impl(replica_id, model, dataset, params)

def main():
    params = params_ad
    gpus_to_use = params.gpus_to_use
    replica_count = len(gpus_to_use)

    print("replica_count: ", replica_count)

    if params.batch_size % replica_count != 0:
        raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus_to_use))

    spawn(train_distributed, args=(replica_count, port, params), nprocs=replica_count, join=True)

if __name__ == '__main__':
    main()
