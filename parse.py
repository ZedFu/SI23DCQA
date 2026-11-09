"""Configuration and runtime overrides shared by the original training entries."""

import argparse
import os
from pathlib import Path

import yaml


class ParserClass:
    def __init__(self, opt):
        self.opt = opt
        self.exp_name = opt['exp']['name']
        self.exp_device = opt['exp']['device']
        self.exp_accelerator = opt['exp'].get('accelerator', 'auto')
        self.save_checkpoint = opt['exp'].get('save_checkpoint', True)
        self.num_workers = opt['train'].get('num_workers', 8)
        for key in ['num_split', 'root', 'ann', 'sample_num', 'size', 'train', 'test']:
            if key in opt['dataset']:
                setattr(self, 'dataset_' + key, opt['dataset'][key])
        self.train_epochs = opt['train']['epochs']
        self.train_bs = opt['train']['batch_size']
        self.model_backbone = opt['model']['backbone']
        for key in ['vl_model', 'sd_model', 'td_model']:
            if key in opt['model']:
                setattr(self, key, opt['model'][key])
        self.op_lr = opt['optimizer']['lr']


def add_runtime_arguments(parser):
    parser.add_argument('--data-root', help='Override dataset root and annotation paths under it.')
    parser.add_argument('--device', help='One CUDA device index (e.g. 0), or cpu.')
    parser.add_argument('--num-workers', type=int, help='DataLoader workers; use 0 for debugging.')
    parser.add_argument('--no-checkpoints', action='store_true', help='Disable saving last.ckpt.')


def load_options(args):
    with open(args.con, encoding='utf-8') as stream:
        opt = yaml.safe_load(stream)
    dataset = opt['dataset']
    original_root = Path(os.path.expandvars(dataset['root'])).expanduser().resolve()
    override = args.data_root or os.environ.get('SI23DCQA_DATA_ROOT')
    dataset['root'] = str(Path(os.path.expandvars(override or dataset['root'])).expanduser().resolve())
    for key in ['ann', 'train', 'test']:
        if key not in dataset:
            continue
        original = Path(os.path.expandvars(dataset[key])).expanduser().resolve()
        if override and original.is_relative_to(original_root):
            dataset[key] = str(Path(dataset['root']) / original.relative_to(original_root))
        else:
            dataset[key] = str(original)
    if args.device:
        if args.device == 'cpu':
            opt['exp'].update(device=1, accelerator='cpu')
        else:
            index = int(args.device)
            if index < 0:
                raise ValueError('--device must be a nonnegative CUDA index or cpu')
            opt['exp'].update(device=[index], accelerator='gpu')
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError('--num-workers must be nonnegative')
        opt['train']['num_workers'] = args.num_workers
    if args.no_checkpoints:
        opt['exp']['save_checkpoint'] = False
    return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--con', default='config/my-all.yml')
    add_runtime_arguments(parser)
    print(yaml.safe_dump(load_options(parser.parse_args()), sort_keys=False))
