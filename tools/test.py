import argparse
import gorilla, os
import torch
from tqdm import tqdm
import numpy as np
from stmn.dataset import build_dataloader, build_dataset
from stmn.model import STMN
from stmn.utils import get_root_logger, save_pred_instances
import json

def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', default=None, type=str, help='directory for output results')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpu_id', type=int, default=[0], nargs='+', help='ids of gpus to use')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_id, num_gpu=args.num_gpus)

    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger(log_file=args.checkpoint.replace('.pth', '.log'))

    model = STMN(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.val, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.val)

    scan_ids, object_ids, ann_ids, pious, spious, gt_pmasks, pred_pmasks = [], [], [], [], [], [], []
    
    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            res = model(batch, mode='predict')
            scan_ids.append(res['scan_id'])
            object_ids.append(res['object_id'])
            ann_ids.append(res['ann_id'])
            pious.append(res['piou'].cpu().numpy())
            spious.append(res['spiou'].cpu().numpy())
            gt_pmasks.append(res['gt_pmask'].cpu().numpy())
            pred_pmasks.append(res['pred_pmask'].sigmoid().cpu().numpy())
            progress_bar.update()
        progress_bar.close()

    logger.info('Evaluate referring segmentation')
    # point-level metrics
    pious = np.stack(pious, axis=0)
    # superpoint-level metrics
    spious = np.stack(spious, axis=0)
    spprecision_half = (spious > 0.5).sum().astype(float) / spious.size
    spprecision_quarter = (spious > 0.25).sum().astype(float) / spious.size
    spmiou = spious.mean()
    logger.info('sp_Acc@25: {:.3f}. sp_Acc@50: {:.3f}. sp_mIOU: {:.3f}.'.format(spprecision_quarter, spprecision_half, spmiou))
    

    with open(os.path.join(cfg.data.val.data_root,"lookup.json"),'r') as load_f:
        # unique为1, multi为0
        unique_multi_lookup = json.load(load_f)
    unique, multi = [], []
    for idx, scan_id in enumerate(scan_ids):
        if unique_multi_lookup[scan_id][str(object_ids[idx])][str(ann_ids[idx])] == 0:
            unique.append(pious[idx])
        else:
            multi.append(pious[idx])
    unique = np.array(unique)
    multi = np.array(multi)
    for u in [0.25, 0.5]:
        logger.info(f'Acc@{u}: \tunique: '+str(round((unique>u).mean(), 4))+' \tmulti: '+str(round((multi>u).mean(), 4))+' \tall: '+str(round((pious>u).mean(), 4)))
    logger.info('mIoU:\t \tunique: '+str(round(unique.mean(), 4))+' \tmulti: '+str(round(multi.mean(), 4))+' \tall: '+str(round(pious.mean(), 4)))
    
    # save output
    if args.out is None:
        output = input('If you want to save the results? (y/n)')
        if output == 'y':
            args.out = os.path.join(os.path.dirname(args.checkpoint), 'results')
        else:
            logger.info('Not saving results.')
            exit()
        
    if args.out:
        logger.info('Saving results...')
        save_pred_instances(args.out, 'pred_instance', scan_ids, object_ids, ann_ids, pred_pmasks)
        logger.info('Done.')

if __name__ == '__main__':
    main()
