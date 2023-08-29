import multiprocessing as mp
import numpy as np
import os
import os.path as osp

from .mask_encoder import rle_decode


def save_single_instance(root, scan_id, object_id, ann_id, pred_pmask):
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    np.savetxt(osp.join(root,'predicted_masks', f'{scan_id}_{object_id}_{ann_id}.txt'), pred_pmask, fmt='%f')


def save_pred_instances(root, name, scan_ids, object_ids, ann_ids, pred_pmasks):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)

    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, object_ids, ann_ids, pred_pmasks))
    pool.close()
    pool.join()


def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt='%d')


def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()
