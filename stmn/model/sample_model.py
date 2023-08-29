import torch, math
import torch.nn as nn
import numpy as np
import sys
import os
import torch.nn.functional as F
if os.path.join(os.getcwd()) not in sys.path:
    sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(os.getcwd(), "lib"))
sys.path.append(os.path.join(os.getcwd(), "lib", 'pointnet2'))

class CrossAttentionMap(nn.Module):
    def __init__(self, q_dim, k_dim, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(q_dim, d_model, bias=False)
        self.key_proj = nn.Linear(k_dim, d_model, bias=False)

    def forward(self, query, key, query_mask=None, key_padding_mask=None):
        """
        Args:
            query: (B, N, D1)
            key: (B, L, D2)
            query_mask: (B, N)
            key_padding_mask: (B, L)
        Returns:
            attn_map: (B, N, L)
        """
        # Apply projection to query and key
        Q = self.query_proj(query)  # (B, N, D_model)
        K = self.key_proj(key)  # (B, L, D_model)
        # Compute attention scores
        scores = torch.einsum('bnd,bld->bnl', Q, K) / math.sqrt(self.d_model)  # (B, N, L)
        # Apply masks
        if query_mask is not None:
            scores = scores.masked_fill(query_mask.unsqueeze(2), float('-inf'))
        # Compute attention map
        attn_map = F.softmax(scores, dim=1)
        return attn_map
    
class SamplingModule(nn.Module):
    """
    Sample object proposal.
    """
    def __init__(self, num_proposal, pc_dim, lang_dim, d_model):
        super().__init__()
        self.num_proposal = num_proposal
        
        self.camap = CrossAttentionMap(pc_dim, lang_dim, d_model)

    def forward(self, pc_feat, lang_feat, pc_mask, lang_mask):
        """
        Args:
            features: (B, N, D)
            lang_feat: (B, L, D)
        """
        # (B, N, L)
        ref_scores = self.camap(pc_feat, lang_feat, pc_mask.bool(), lang_mask.bool()) # (B, N, L)
        ref_scores = ref_scores * (~lang_mask.unsqueeze(1)).float() # (B, N, L)
        ref_scores = ref_scores.sum(-1)   # [B, N]
        # print(ref_scores)
        if ref_scores.shape[1] < self.num_proposal:
            sample_inds = torch.Tensor(list(range(ref_scores.shape[1]))).int().unsqueeze(0).repeat(ref_scores.shape[0], 1).cuda()
        else:
            sample_inds = torch.topk(ref_scores, self.num_proposal)[1].int() # (bsz, num_proposal)

        return sample_inds, ref_scores

if __name__=='__main__':
    from easydict import EasyDict
    data_dict = {'point_clouds': torch.rand(16,3000,6).cuda()}
    args = EasyDict({
        'kps_fusion_dim': 256
    })
    dks_net = SamplingModule(num_proposal=512, feat_dim=32, lang_dim=256, d_model=256).cuda()
    pc_xyz = torch.rand(16, 3000, 3).cuda()
    pc_feat = torch.rand(16, 32, 3000).cuda()
    lang_feat = torch.rand(16, 256).cuda()
    data_dict['lang_hidden'] = lang_feat
    out_dict, xyz, feat = dks_net(pc_xyz, pc_feat, data_dict)
    print(xyz.shape, feat.shape)
    for key in sorted(out_dict.keys()):
        print(key, '\t', out_dict[key].shape)