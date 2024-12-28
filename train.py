import os
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

from config import Config
from trainer import train_imagenet

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    dist.destroy_process_group()

def main_worker(rank, world_size, config):
    if config.distributed:
        setup_distributed(rank, world_size)
    
    train_imagenet(config)
    
    if config.distributed:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=256,
                      help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='number of epochs to train')
    parser.add_argument('--workers', type=int, default=8,
                      help='number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.1,
                      help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='weight decay')
    parser.add_argument('--no-distributed', action='store_true',
                      help='disable distributed training')
    parser.add_argument('--no-amp', action='store_true',
                      help='disable automatic mixed precision')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                      help='path to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                      help='enable wandb logging')
    
    args = parser.parse_args()
    
    # Create config from args
    config = Config()
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.num_workers = args.workers
    config.learning_rate = args.lr
    config.momentum = args.momentum
    config.weight_decay = args.weight_decay
    config.distributed = not args.no_distributed
    config.use_amp = not args.no_amp
    config.save_dir = args.save_dir
    config.use_wandb = args.wandb
    
    if config.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(main_worker,
                args=(world_size, config),
                nprocs=world_size,
                join=True)
    else:
        main_worker(0, 1, config)

if __name__ == '__main__':
    main() 