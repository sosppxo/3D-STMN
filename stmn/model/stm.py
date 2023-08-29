import torch
import torch.nn as nn
from .sample_model import SamplingModule
from torch_scatter import scatter_max, scatter_mean, scatter
# modified torch multihead attention
from ..torch.nn import MultiheadAttention
# graph
from .graph.graph_transformer_net import GraphTransformerNet
from .graph.layers.graph_transformer_edge_layer import GraphTransformerLayer, GraphTransformerSubLayer

class DDI(nn.Module):

    def __init__(
            self,
            hidden_dim,
            out_dim,
            n_heads,
            dropout=0.0,
            layer_norm=True, 
            batch_norm=False, 
            residual=True, 
            use_bias=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.graph_attn = GraphTransformerSubLayer(hidden_dim, out_dim, n_heads, dropout, layer_norm, batch_norm)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def graph2batch(self, batched_graph):
        node_num = batched_graph.batch_num_nodes()
        batch_offsets = torch.cat([torch.tensor((0,), dtype=torch.int).to(batched_graph.device), node_num.cumsum(0).int()], dim=0)
        batch_data, batch_masks = self.get_batches(batched_graph.ndata['h'], batch_offsets)
        return batch_data, batch_masks
    
    def batch2graph(self, batch_data, batch_masks):
        B = batch_data.shape[0]
        batch_x = []
        for i in range(B):
            batch_x.append(batch_data[i, (~batch_masks[i])])
        batch_x = torch.cat(batch_x, dim=0)
        return batch_x
    
    def forward(self, x, x_mask, batch_g, batch_x, batch_e, pe=None, cat='parallel'):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        # parallel
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        sa_output, _ = self.self_attn(q, k, x, key_padding_mask=x_mask)

        # graph-attention
        batch_x, batch_e = self.graph_attn(batch_g, batch_x, batch_e)
        batch_g.ndata['h'] = batch_x
        batch_g.edata['e'] = batch_e

        # transform batched graph to batched tensor
        ga_output, _ = self.graph2batch(batch_g)
        ga_output = torch.cat([ga_output, torch.zeros(B, x_mask.shape[1]-ga_output.shape[1], ga_output.shape[-1]).to(ga_output.device)], dim=1)

        # residual connection
        output = self.dropout(sa_output + ga_output) + x
        output = self.norm(output)

        return output, batch_e

class SWA(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[1], k.shape[1])
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output, output_weight, src_weight # (b, n_q, d_model), (b, n_q, n_v)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output
    
class STM(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """

    def __init__(
        self,
        num_layer=6,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        sampling_module=None,
        kernel='top1',
        global_feat='mean',
        graph_params=None,
        bidirectional=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        H = 768
        self.lang_proj = nn.Linear(H, 256)
        if sampling_module is not None:
            self.sampling_module = SamplingModule(**sampling_module)
        else:
            self.sampling_module = None

        self.lap_pos_enc = graph_params['lap_pos_enc']
        if graph_params['lap_pos_enc']:
            pos_enc_dim = graph_params['pos_enc_dim']
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, d_model)

        self.embedding_root = nn.Linear(1, d_model)
        if bidirectional:
            self.embedding_e = nn.Embedding(2*graph_params['num_bond_type'], d_model)
        else:
            self.embedding_e = nn.Embedding(graph_params['num_bond_type'], d_model)

        # DDI and SWA
        self.ddi_layers = nn.ModuleList([])
        self.ddi_ffn_layers = nn.ModuleList([])
        self.swa_layers = nn.ModuleList([])
        self.swa_ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.ddi_layers.append(DDI(graph_params['hidden_dim'], graph_params['out_dim'], graph_params['n_heads'], graph_params['dropout'], graph_params['layer_norm'], graph_params['batch_norm'], graph_params['residual']))
            self.ddi_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.swa_layers.append(SWA(d_model, nhead, dropout))
            self.swa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.kernel = kernel
        self.global_feat = global_feat
    
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)

            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, mask_feats, batch_mask):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks
    
    def avg_lang_feat(self, lang_feats, lang_masks):
        lang_len = lang_masks.sum(-1)
        lang_len = lang_len.unsqueeze(-1)
        lang_len[torch.where(lang_len == 0)] = 1
        return (lang_feats * ~lang_masks.unsqueeze(-1).expand_as(lang_feats)).sum(1) / lang_len

    def prediction_head(self, query, mask_feats, batch_mask):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_scores, pred_masks, attn_masks

    def graph2batch(self, batched_graph):
        node_num = batched_graph.batch_num_nodes()
        batch_offsets = torch.cat([torch.tensor((0,), dtype=torch.int).to(batched_graph.device), node_num.cumsum(0).int()], dim=0)
        batch_data, batch_masks = self.get_batches(batched_graph.ndata['h'], batch_offsets)
        return batch_data, batch_masks
    
    def batch2graph(self, batch_data, batch_masks):
        B = batch_data.shape[0]
        batch_x = []
        for i in range(B):
            batch_x.append(batch_data[i, (~batch_masks[i])])
        batch_x = torch.cat(batch_x, dim=0)
        return batch_x
    
    def forward_iter_pred(self, x, batch_offsets, batched_graph, lang_feats=None, lang_masks=None):
        """
        x [B*M, inchannel]
        """
        # process the graph feats
        batched_graph = batched_graph.to(x.device)
        try:
            batch_lap_pos_enc = batched_graph.ndata['lap_pos_enc'].to(x.device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(x.device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None
            
        # prepare for DDI
        lang_feats = self.lang_proj(lang_feats) 
        lang_feats = lang_feats[:, 1:, :] 
        lang_len = lang_masks.sum(-1) - 2 
        lang_masks = torch.arange(lang_feats.shape[1])[None, :].to(lang_feats.device) < lang_len[:, None]  
        lang_masks = ~lang_masks 
        query = lang_feats

        root_embedding = self.embedding_root(torch.tensor((0,)).float().to(x.device)).unsqueeze(0)
        # every sentence has a root node, there are B sentences
        assert (lang_len+1 - batched_graph.batch_num_nodes()).sum() == 0
        batch_x = torch.cat([torch.cat([root_embedding, query[i][~(lang_masks[i].bool())]], dim=0) for i in range(query.shape[0])], dim=0)
        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(batch_lap_pos_enc.float()) 
            batch_x = batch_x + h_lap_pos_enc
        batched_graph.ndata['h'] = batch_x
        
        # add root node
        query, _ = self.graph2batch(batched_graph)
        query = torch.cat([query, torch.zeros(lang_feats.shape[0], lang_feats.shape[1]-query.shape[1], query.shape[-1]).to(query.device)], dim=1)
        lang_masks = torch.arange(lang_feats.shape[1])[None, :].to(lang_feats.device) <= lang_len[:, None] 
        lang_masks = ~lang_masks 

        batch_e = batched_graph.edata['feat'].to(x.device)
        batch_e = self.embedding_e(batch_e)
        batched_graph.edata['e'] = batch_e

        # prepare for SWA
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)
        prediction_masks = []
        prediction_scores = []
        B = len(batch_offsets) - 1
        
        # 0-th prediction
        pred_scores, pred_masks, attn_masks = self.prediction_head(self.avg_lang_feat(query, lang_masks).unsqueeze(1), mask_feats, batch_mask)
        _, _, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)

        sample_inds = None
        ref_scores = None

        # sampling
        if hasattr(self, 'sampling_module'):
            if self.global_feat == 'mean':
                global_feats = scatter_mean(inst_feats, batch_mask.long(), dim=1)[:,0,:]
            else:
                global_feats = scatter_max(inst_feats, batch_mask.long(), dim=1)[0][:,0,:]
            sample_inds, ref_scores = self.sampling_module(inst_feats, query, batch_mask, lang_masks)
            sample_inds = sample_inds.long()
            inst_feats = torch.gather(inst_feats, dim=1, index=sample_inds.unsqueeze(-1).repeat(1,1,inst_feats.shape[-1]))
            batch_mask_sampled = torch.gather(batch_mask, dim=1, index=sample_inds)
            attn_masks = torch.gather(attn_masks, dim=2, index=sample_inds.unsqueeze(1).repeat(1,attn_masks.shape[1],1))
            global_feats_mask = torch.zeros(B, 1).bool().to(batch_mask_sampled.device)
            batch_mask_sampled = torch.cat([batch_mask_sampled, global_feats_mask], dim=1)

        # multi-round
        for i in range(self.num_layer):
            # DDI
            batch_x = self.batch2graph(query, lang_masks)
            query, batch_e = self.ddi_layers[i](query, lang_masks, batched_graph, batch_x, batch_e)
            query = self.ddi_ffn_layers[i](query)

            # SWA
            if hasattr(self, 'sampling_module'):
                attn_masks = torch.cat([attn_masks, global_feats_mask.unsqueeze(1).repeat(1,attn_masks.shape[1],1)], dim=2)
                query, _, src_weight = self.swa_layers[i](torch.cat([inst_feats, global_feats.unsqueeze(1)], dim=1), query, batch_mask_sampled, attn_masks)
            else:
                query, _, src_weight = self.swa_layers[i](inst_feats, query, batch_mask, attn_masks)
            query = self.swa_ffn_layers[i](query)

            if self.kernel=='top1':
                src_weight = src_weight.softmax(1)
                src_weight = torch.where(torch.isnan(src_weight), torch.zeros_like(src_weight), src_weight)
                q_score = (src_weight*~lang_masks.unsqueeze(-1)).sum(-1) # [B, N_q]
                _, q_idx = q_score.topk(1, dim=-1) # [B, 1]
                pred_scores, pred_masks, _ = self.prediction_head(query.gather(dim=1, index=q_idx.unsqueeze(-1).repeat(1, 1, query.size(-1))), mask_feats, batch_mask)
            _, _, attn_masks = self.prediction_head(query, mask_feats, batch_mask)

            if hasattr(self, 'sampling_module'):
                attn_masks = torch.gather(attn_masks, dim=2, index=sample_inds.unsqueeze(1).repeat(1,attn_masks.shape[1],1))
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)

        return {
            'masks':
            pred_masks,
            'batch_mask':
            batch_mask,
            'scores':
            pred_scores,
            'sample_inds':
            sample_inds, # [B, K]
            'ref_scores':
            ref_scores, # [B, M]
            'aux_outputs': [{
                'masks': a,
                'scores': b
            } for a, b in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
            )],
        }

    def forward(self, x, batch_offsets, batched_graph, lang_feats=None, lang_masks=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, batch_offsets, batched_graph, lang_feats, lang_masks)
        else:
            raise NotImplementedError
