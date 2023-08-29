import glob, json, os
import math
import numpy as np
import os.path as osp
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from typing import Dict, Sequence, Tuple, Union
from tqdm import tqdm
from transformers import BertTokenizer
from gorilla import is_main_process
import dgl
from scipy import sparse as sp
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    # A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g

class ScanNetDataset_sample_graph_edge(Dataset):

    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 mode=4,
                 with_elastic=True,
                 use_xyz=True,
                 logger=None,
                 max_des_len=78,
                 graph_pos_enc_dim=5,
                 bidirectional=False,):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.use_xyz = use_xyz
        self.logger = logger
        self.max_des_len = max_des_len
        self.bidirectional = bidirectional
        self.depend2id = torch.load(os.path.join(self.data_root, 'dependency_map.pth'))['depend2id']
        self.id2depend = torch.load(os.path.join(self.data_root, 'dependency_map.pth'))['id2depend']
        self.graph_pos_enc_dim = graph_pos_enc_dim
        self.filenames = self.get_graph_filenames()
        self.sp_filenames = self.get_sp_filenames()
        
        # load scanrefer
        if self.prefix == 'train':
            self.scanrefer = json.load(open(os.path.join(self.data_root, 'ScanRefer', 'ScanRefer_filtered_train.json')))
            if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
        elif self.prefix == 'val':
            self.scanrefer = json.load(open(os.path.join(self.data_root, 'ScanRefer', 'ScanRefer_filtered_val.json')))
            if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
        else:
            raise ValueError('ScanRefer only support train and val split, not support %s' % self.prefix)
        # load lang
        self.load_lang()
        # main(instance seg task) with others
        self.type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
        self.nyu40id2class = self._get_nyu40id2class()
        self.sem2nyu = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


    def _get_type2class_all(self):
        lines = [line.rstrip() for line in open(os.path.join(self.data_root, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        type2class = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                class_id = np.where(self.nyu40ids == nyu40_id)[0][0]
                type2class[nyu40_name] = class_id
        return type2class
    
    def _get_nyu40id2class(self):
        lines = [line.rstrip() for line in open(os.path.join(self.data_root, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(self.type2class.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = self.type2class["others"]
                else:
                    nyu40ids2class[nyu40_id] = self.type2class[nyu40_name]
        return nyu40ids2class

    def load_lang(self):
        lang = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = int(data["object_id"])
            ann_id = int(data["ann_id"])

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
            # load tokens temporarily, and you can extract then load embeddings or features if needed in the future
            lang[scene_id][object_id][ann_id] = data["token"]
        self.lang = lang

    def get_graph_filenames(self):
        if not os.path.exists(osp.join(self.data_root, 'features', self.prefix, 'graph')):
            os.makedirs(osp.join(self.data_root, 'features', self.prefix, 'graph'))
        filenames = glob.glob(osp.join(self.data_root, 'features', self.prefix, 'graph', '*' + str(self.max_des_len).zfill(3) + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
 
        graphs = {}
        if is_main_process():
            print('loading ' + self.prefix + ' graphs...')
        for filename in tqdm(filenames):
            # source graph filename
            graph_fn = filename
            # dgl filename
            dgl_fn = filename.replace(self.suffix, '.dgl')
            if not osp.exists(graph_fn):
                raise ValueError('Graph file not found: ' + graph_fn)
            
            graph = torch.load(graph_fn)
            heads = graph['heads']
            assert heads[0] == 0, 'ROOT node must be at the beginning'
            tails = graph['tails']
            relations = graph['relations']
            words = graph['words']
            words = [words[i-1] for i in sorted(list(set(tails)))]

            # build dgl graph and save
            if not osp.exists(dgl_fn):
                g = dgl.graph((tails, heads))
                # ROOT node
                token_id = [tokenizer.vocab_size]
                # words without cls token and sep token
                token_id += tokenizer.encode(words, add_special_tokens=False)
                assert len(token_id)==g.num_nodes()
                g.ndata['feat'] = torch.tensor(token_id)
                # edge feat
                relation_id = [self.depend2id[relation] for relation in relations]
                assert len(relation_id)==g.num_edges()
                g.edata['feat'] = torch.tensor(relation_id)
                laplacian_positional_encoding(g, self.graph_pos_enc_dim)
                dgl.save_graphs(dgl_fn, g)

            graphs.update({osp.basename(filename): 
                            {'graph_file': dgl_fn,
                            'tokens': words,
                            }
                        })
        self.graphs = graphs
        return filenames

    def get_sp_filenames(self):
        filenames = glob.glob(osp.join(self.data_root, 'scannetv2', self.prefix, '*' + '_refer.pth'))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        return filenames
        
    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label
        
    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz_middle = self.data_aug(xyz, True, True, True)
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        xyz, valid_idxs = self.crop(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        r"""
        crop the point cloud to reduce training complexity

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud to be cropped

        Returns:
            Union[np.ndarray, np.ndarray]: processed point cloud and boolean valid indices
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag

    def get_cropped_inst_label(self, instance_label: np.ndarray, valid_idxs: np.ndarray) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label
    
    def get_ref_mask(self, instance_label, superpoint, object_id):
        ref_lbl = instance_label == object_id
        gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index: int) -> Tuple:
        filename = self.filenames[index]
        # scene0000_00_000_000.pth
        scan_id = osp.basename(filename)[:12]
        object_id = int(osp.basename(filename)[13:16])
        ann_id = int(osp.basename(filename)[17:20])
        lang_tokens = self.lang[scan_id][object_id][ann_id]

        # load point cloud
        for fn in self.sp_filenames:
            if scan_id in fn:
                sp_filename = fn
                break
        data = self.load(sp_filename)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = data


        context_label = set()
        for word in lang_tokens:
            if word in self.type2class.keys() and word != 'others':
                context_label.add(self.type2class[word])
        point_context_mask = np.zeros(instance_label.shape[0]) - 1
        for i_instance in np.unique(instance_label):            
            # find all points belong to that instance
            ind = np.where(instance_label == i_instance)[0]
            # find the semantic label            
            if int(semantic_label[ind[0]])>=0:
                nyu_id = int(self.sem2nyu[int(semantic_label[ind[0]])])
                if nyu_id in self.nyu40ids and self.nyu40id2class[nyu_id] in context_label:
                    point_context_mask[ind] = 1
        point_ref_mask = np.zeros(instance_label.shape[0])
        # assert len(context_label)==0 or point_context_mask.max()>0, 'no context points'
        point_ref_mask[point_context_mask > 0] = 0.5
        point_ref_mask[instance_label == object_id] = 1

        g = dgl.load_graphs(self.graphs[osp.basename(filename)]['graph_file'])[0][0]
        lang_tokens = self.graphs[osp.basename(filename)]['tokens']

        if self.bidirectional:
            try:
                bg = dgl.to_bidirected(g)
                bg.edata['feat'] = torch.cat([g.edata['feat'], g.edata['feat']+40], dim=0)
                bg.ndata['feat'] = g.ndata['feat']
                bg.ndata['lap_pos_enc'] = g.ndata['lap_pos_enc']
                g = bg
            except:
                if is_main_process():
                    self.logger.info('multigraph: '+scan_id+str(object_id).zfill(3)+str(ann_id).zfill(3))

        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        semantic_label = torch.from_numpy(semantic_label).long()
        instance_label = torch.from_numpy(instance_label).long()
        gt_pmask, gt_spmask = self.get_ref_mask(instance_label, superpoint, object_id)
        point_ref_mask = torch.from_numpy(point_ref_mask).float()
        sp_ref_mask = torch_scatter.scatter_mean(point_ref_mask, superpoint, dim=-1)

        return ann_id, scan_id, coord, coord_float, feat, superpoint, object_id, gt_pmask, gt_spmask, sp_ref_mask , g, lang_tokens
    
    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        ann_ids, scan_ids, coords, coords_float, feats, superpoints, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, batched_graph, lang_tokenss, lang_masks, lang_words = [], [], [], [], [], [], [], [], [], [], [], [], [], []
        batch_offsets = [0]
        superpoint_bias = 0

        for i, data in enumerate(batch):
            ann_id, scan_id, coord, coord_float, feat, src_superpoint, object_id, gt_pmask, gt_spmask, sp_ref_mask, g, lang_tokens = data
            
            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)

            ann_ids.append(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            
            object_ids.append(object_id)
            
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            sp_ref_masks.append(sp_ref_mask)
            
            token_dict = tokenizer.encode_plus(lang_tokens, add_special_tokens=True, truncation=True, max_length=self.max_des_len+2, padding='max_length', return_attention_mask=True,return_tensors='pt',)
            assert len(lang_tokens) == g.num_nodes()-1
            lang_words.append(lang_tokens)
            lang_tokenss.append(token_dict['input_ids']) 
            lang_masks.append(token_dict['attention_mask'])

            batched_graph.append(g)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)
        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        lang_tokenss = torch.cat(lang_tokenss, 0)
        lang_masks = torch.cat(lang_masks, 0).int()
        # merge all scan in batch
        batched_graph = dgl.batch(batched_graph)

        return {
            'ann_ids': ann_ids,
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'object_ids': object_ids,
            'gt_pmasks': gt_pmasks,
            'gt_spmasks': gt_spmasks,
            'sp_ref_masks': sp_ref_masks,
            'batched_graph': batched_graph,
            'lang_tokenss': lang_tokenss,
            'lang_masks': lang_masks,
        }
