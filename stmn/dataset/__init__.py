from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .scannetv2 import ScanNetDataset_sample_graph_edge

__all__ = ['ScanNetDataset_sample_graph_edge', 'build_dataset', 'build_dataloader']


def build_dataset(data_cfg, logger):
    assert 'type' in data_cfg
    _data_cfg = data_cfg.copy()
    _data_cfg['logger'] = logger
    data_type = _data_cfg.pop('type')
    if data_type == 'scannetv2_sample_graph_edge':
        return ScanNetDataset_sample_graph_edge(**_data_cfg)
    else:
        raise ValueError(f'Unknown {data_type}')


def build_dataloader(dataset, batch_size=1, num_workers=1, training=True, dist=False, persistent_workers=True):
    shuffle = training
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist else None
    if sampler is not None:
        shuffle = False
    if training:
        # 读取预处理特征的话需要bs=64
        # if isinstance(dataset, ScanNetDataset_sp): assert batch_size == 64     
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            persistent_workers=persistent_workers)
    else:
        assert batch_size == 1
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=True,
            persistent_workers=persistent_workers)
