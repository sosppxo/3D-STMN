import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
import functools

from stmn.utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion, get_iou
from .stm import STM
from transformers import BertModel

class STMN(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        sampling_module=None,
        stm=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
    ):
        super().__init__()

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool

        self.decoder_param = stm

        # bert encoder
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')

        self.sampling_module = sampling_module
        # stm
        self.stm = STM(**stm, sampling_module=sampling_module, in_channel=media)
        # criterion
        self.criterion = Criterion(**criterion)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        self.init_weights()
        for module in fix_module:
            if '.' in module:
                module, params = module.split('.')
                module = getattr(self, module)
                params = getattr(module, params)
                for param in params.parameters():
                    param.requires_grad = False
            else:
                module = getattr(self, module)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super(STMN, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, ann_ids, scan_ids, sp_feats, superpoints, batch_offsets, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks=None, batched_graph=None, lang_feats=None, lang_masks=None):
        out = self.stm(sp_feats, batch_offsets, batched_graph,  lang_feats, lang_masks) # sent_kernel [B, 1, 256]
        if self.sampling_module is not None:
            loss, loss_dict = self.criterion(out, gt_pmasks, gt_spmasks, sp_ref_masks)
        else:
            loss, loss_dict = self.criterion(out, gt_pmasks, gt_spmasks, None)
        return loss, loss_dict
    
    @cuda_cast
    def predict(self, ann_ids, scan_ids, sp_feats, superpoints, batch_offsets, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks=None, batched_graph=None, lang_feats=None, lang_masks=None):
        out = self.stm(sp_feats, batch_offsets, batched_graph,  lang_feats, lang_masks) # sent_kernel [B, 1, 256]
        ret = self.predict_by_feat(scan_ids, object_ids, ann_ids, out, superpoints, gt_pmasks, gt_spmasks)

        return ret
    
      
    def predict_by_feat(self, scan_ids, object_ids, ann_ids, out, superpoints, gt_pmasks, gt_spmasks):
        # B is 1 when predecit
        gt_pmask = gt_pmasks[0]
        gt_spmask = gt_spmasks[0]
        pred_spmask = out['masks'][-1].squeeze()
        spiou = get_iou(pred_spmask, gt_spmask)

        pred_pmask = pred_spmask[superpoints]
        piou = get_iou(pred_pmask, gt_pmask)

        return dict(scan_id=scan_ids[0], object_id=object_ids[0], ann_id=ann_ids[0], piou=piou, spiou=spiou, gt_pmask=gt_pmask, pred_pmask=pred_pmask)

    def extract_feat(self, x, superpoints, v2p_map):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        # superpoint pooling
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        return x
    