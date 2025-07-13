# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
EventSem model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from EventSem.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from EventSem.position_encoding import build_position_encoding, PositionEmbeddingSine
import math
from nncore.nn import build_model as build_adapter
from blocks.generator import PointGenerator

from scipy.optimize import linear_sum_assignment

import logging

logger = logging.getLogger(__name__)

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start+len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start

def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a==b:
            res.append(True)
        else:
            res.append(False)
    return res

class ConfidenceScorer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_conv_layers=1, num_mlp_layers=3):
        super(ConfidenceScorer, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for i in range(num_conv_layers):
            if i == 0:
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            else:
                self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            self.activations.append(nn.ReLU(inplace=False))
        
        self.fc = MLP(out_channels, out_channels // 2, 1, num_layers=num_mlp_layers)
    
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        
        for conv, activation in zip(self.convs, self.activations):
            x = conv(x)
            x = activation(x)
        
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.fc(x)
        
        return x

class EventSem(nn.Module):
    """ EventSem. """

    def __init__(self, transformer, position_embed, txt_position_embed, n_input_proj, input_dropout, txt_dim, vid_dim, aud_dim=0, use_txt_pos=False,
                strides=(1, 2, 4, 8),
                buffer_size=2048,
                max_num_moment=50,
                merge_cls_sal=True,
                pyramid_cfg=None,
                pooling_cfg=None,
                coord_head_cfg=None,
                args=None):
        """ Initializes the model."""
        super().__init__()
        self.args=args
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.PositionEmbeddingSine = PositionEmbeddingSine(hidden_dim, normalize=True)
        
        # input projection
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # set up dummy token
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.use_txt_pos = use_txt_pos
        self.dummy_rep_token = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        normalize_before = False
        input_txt_sa_proj = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        txtproj_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, args.dummy_layers, txtproj_encoder_norm)

        # build muti-scale pyramid
        self.pyramid = build_adapter(pyramid_cfg, hidden_dim, strides)

        self.pooling = build_adapter(pooling_cfg, hidden_dim)
        self.conf_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.class_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.coef = nn.Parameter(torch.ones(len(strides)))
        self.coord_head = build_adapter(coord_head_cfg, hidden_dim, 2)
        self.generator = PointGenerator(strides, buffer_size)
        self.max_num_moment = max_num_moment
        self.merge_cls_sal = merge_cls_sal
        self.args = args
        self.x = nn.Parameter(torch.tensor(0.5))
        
        self.event_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        self.saliency_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.saliency_token, std=0.02)
        self.saliency_attn = nn.MultiheadAttention(hidden_dim, 8, dropout=0.1)
        self.saliency_norm = nn.LayerNorm(hidden_dim)
        self.saliency_dropout = nn.Dropout(0.1)
        self.refine_conv_net = nn.Sequential(
            nn.Conv1d(hidden_dim + 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 2, kernel_size=1)  # 输出2个通道对应两个边界偏移
        )
        self.n_input_proj = self.args.n_semantic_proj
        layers = []
        for i in range(self.args.n_semantic_proj):
            relu_enabled = True
            if i == self.args.n_semantic_proj - 1:
                relu_enabled = False
# qvhighlight 384
            input_dim = 300 if i == 0 else hidden_dim
            layers.append(
                LinearLayer(input_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_enabled)
            )

        self.input_sematic_txt_proj = nn.Sequential(*layers)
        self.input_src_txt_proj = nn.Sequential(*[
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj]) 
        
        self.gate = nn.Parameter(torch.tensor(self.args.gate))
        
        self.sim_sharpness = nn.Parameter(torch.tensor(5.0))
    def generate_event_prior_mask(self, pseudo_event_spans, event_query_sim, video_length):
        """
        Args:
            pseudo_event_spans:  [batch_size] 每个元素是形状为 [num_spans, 2] 的张量
            event_query_sim:  [batch_size, max_spans]
            video_length: 
            
        Returns:
            normalized_mask: [batch_size, video_length]
        """
        batch_size = len(pseudo_event_spans)
        device = event_query_sim.device
        
        event_prior_mask = torch.zeros(batch_size, video_length, device=device)
        
        threshold_base = torch.sigmoid(torch.tensor(self.args.event_sim_threshold,device=device))
        
        positions = torch.arange(video_length, device=device).float() / video_length  # [0, 1]范围内的位置
        
        for b in range(batch_size):
            num_spans = min(pseudo_event_spans[b].shape[0], event_query_sim.shape[1])
            if num_spans == 0:
                continue
            
            batch_sim = event_query_sim[b, :num_spans]
    
            mean_sim = batch_sim.mean()
            std_sim = batch_sim.std() + 1e-6 
            
            scale_factor = (threshold_base * 2) - 1 
            adaptive_threshold = mean_sim + scale_factor * std_sim
            
            batch_mask = torch.zeros(video_length, device=device)
            
            for i in range(num_spans):
                sim_score = event_query_sim[b, i]
                sim_weight = torch.sigmoid((sim_score - adaptive_threshold) * self.args.sim_sharpness)
                if sim_weight < 0.05:
                    continue
                
                center = pseudo_event_spans[b][i, 0]
                width = max(pseudo_event_spans[b][i, 1], 0.05)  # 确保最小宽度
                
                rel_pos = torch.abs(positions - center) / (width / 2)
                
                gaussian_weight = torch.exp(-2 * rel_pos**2)
                final_weight = gaussian_weight * sim_weight
                
                batch_mask = torch.maximum(batch_mask, final_weight)
            
            event_prior_mask[b] = batch_mask
        
        normalized_mask = torch.zeros_like(event_prior_mask)
        for b in range(batch_size):
            max_val = event_prior_mask[b].max()
            if max_val > 0:
                normalized_mask[b] = event_prior_mask[b] / max_val
            else:
                normalized_mask[b] = event_prior_mask[b]
        
        return normalized_mask
    
    def compute_event_query_similarity(self, event_features, query_emb, valid_spans_mask=None):
        """
        Calculate the similarity between event features and query text
        
        Args:
            event_features: [batch_size, num_spans, hidden_dim]
            query_emb:  [batch_size, L_txt, hidden_dim]
            valid_spans_mask: [batch_size, num_spans]
            
        Returns:
            similarity: [batch_size, num_spans]
        """
        _, L_txt, _ = query_emb.shape
        
        if L_txt == 0:
            raise ValueError("Query embedding length is zero.")
        
        query_pooled = torch.mean(query_emb, dim=1)  # [batch_size, hidden_dim]
        
        event_proj = self.event_proj(event_features)  # [batch_size, num_spans, hidden_dim]
        query_proj = self.query_proj(query_pooled).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        event_norm = torch.norm(event_proj, dim=2, keepdim=True)
        query_norm = torch.norm(query_proj, dim=2, keepdim=True)
        event_norm = torch.clamp(event_norm, min=1e-6)
        query_norm = torch.clamp(query_norm, min=1e-6)
        event_normalized = event_proj / event_norm
        query_normalized = query_proj / query_norm
        similarity = torch.bmm(event_normalized, query_normalized.transpose(1, 2)).squeeze(2)  # [batch_size, num_spans]
        
        if valid_spans_mask is not None:
            similarity = similarity * valid_spans_mask.float()
            if similarity.sum() == 0:
                similarity += 1e-6 
        
        min_sim = similarity.min(dim=1, keepdim=True)[0]
        max_sim = similarity.max(dim=1, keepdim=True)[0]
        range_sim = torch.clamp(max_sim - min_sim, min=1e-8)
        normalized_sim = (similarity - min_sim) / range_sim
        normalized_sim[range_sim.expand_as(normalized_sim) == 0] = 0.5
        normalized_sim = normalized_sim + 1e-6
        
        return normalized_sim
    
    def extract_event_features(self, src_vid, pseudo_event_spans=None, pseudo_event_spans_used=None, pseudo_event_spans_mask=None):
        """
        Highly optimized event feature extraction function using batch and vectorization operations
        
        Args: 
        src_vid: video features [batch_size, seq_len, hidden_dim] 
        pseudo_event_spans: normalized event boundaries [batch_size, max_spans, 2], each row is [center, width] 
        pseudo_event_ spans_used: non-normalized event boundaries [batch_size, max_spans, 2], each line is [start_idx, end_idx] 
        pseudo_event_spans_mask: mask indicating valid events [batch_size, max_spans]
                    
        Returns: 
        event_features: extracted event features [batch_size, max_spans, hidden_dim] 
        valid_spans_mask: mask indicating valid features [batch_size, max_spans]
        """
        batch_size, seq_len, hidden_dim = src_vid.shape
        device = src_vid.device
        
        max_allowed_spans = self.args.max_event_spans
        if pseudo_event_spans is not None:
            max_allowed_spans = min(max_allowed_spans, pseudo_event_spans.shape[1])
        elif pseudo_event_spans_used is not None:
            max_allowed_spans = min(max_allowed_spans, pseudo_event_spans_used.shape[1])
        else:
            raise ValueError("Both pseudo_event_spans and pseudo_event_spans_used are None. At least one must be provided.")
        
        event_features = torch.zeros(batch_size, max_allowed_spans, hidden_dim, device=device)
        valid_spans_mask = torch.zeros(batch_size, max_allowed_spans, dtype=torch.bool, device=device)
        
        if pseudo_event_spans_mask is None:
            pseudo_event_spans_mask = torch.ones(batch_size, max_allowed_spans, dtype=torch.bool, device=device)
        else:
            pseudo_event_spans_mask = pseudo_event_spans_mask[:, :max_allowed_spans]
        
        all_start_idx = torch.zeros(batch_size, max_allowed_spans, dtype=torch.long, device=device)
        all_end_idx = torch.zeros(batch_size, max_allowed_spans, dtype=torch.long, device=device)
        
        if pseudo_event_spans_used is not None:
            
            spans_used = pseudo_event_spans_used[:, :max_allowed_spans]
            all_start_idx = spans_used[:, :, 0].long()
            all_end_idx = spans_used[:, :, 1].long()
        else:
            
            centers = pseudo_event_spans[:, :max_allowed_spans, 0]
            widths = torch.max(
                pseudo_event_spans[:, :max_allowed_spans, 1],
                torch.tensor(1.0 / seq_len, device=device)
            )
            
            half_widths = widths / 2
            start_indices = ((centers - half_widths) * seq_len).floor().long()
            end_indices = ((centers + half_widths) * seq_len).ceil().long()
            
            all_start_idx = torch.clamp(start_indices, min=0, max=seq_len-1)
            all_end_idx = torch.clamp(end_indices, min=0, max=seq_len-1)
        
        valid_indices = (all_start_idx < seq_len) & (all_end_idx >= 0) & (all_start_idx <= all_end_idx)
        valid_spans_mask = pseudo_event_spans_mask & valid_indices
        
        for b in range(batch_size):
            
            vid_feats = src_vid[b]  # [seq_len, hidden_dim]
            
            curr_valid_mask = valid_spans_mask[b]  # [max_spans]
            
            valid_indices = torch.nonzero(curr_valid_mask, as_tuple=True)[0]
            
            for idx in valid_indices:
                start_idx = all_start_idx[b, idx]
                end_idx = all_end_idx[b, idx]
                
                span_features = vid_feats[start_idx:end_idx+1]  # [span_len, hidden_dim]
                
                if span_features.size(0) > 0:
                    avg_feature = torch.mean(span_features, dim=0)  # [hidden_dim]
                    norm_feature = F.normalize(avg_feature, p=2, dim=0)  # [hidden_dim]
                    event_features[b, idx] = norm_feature
        
        return event_features, valid_spans_mask

    def generate_pseudo_event(self, src_vid, src_vid_mask, targets):
        """
        Highly optimized pseudo-event generation function to maximize parallelism while keeping results constant
        
        Args:
            src_vid: Video features [batch_size, seq_len, hidden_dim]
            src_vid_mask: Video feature mask [batch_size, seq_len]
            targets: Target data

        Returns:
            pseudo_event_spans: List of tensors with normalized event spans [center, width]
            pseudo_event_spans_used: List of tensors with unnormalized event spans [start, end]
        """
        bsz, L_src, _ = src_vid.size()
        device = src_vid.device
        max_spans = self.args.max_event_spans
        pseudo_event_spans = torch.zeros(bsz, max_spans, 2, device=device)
        pseudo_event_spans_used = torch.zeros(bsz, max_spans, 2, device=device)
        pseudo_event_spans_mask = torch.zeros(bsz, max_spans, dtype=torch.bool, device=device)

        norm_factor = torch.norm(src_vid, dim=2, keepdim=True) + 1e-8
        norm_vid = src_vid / norm_factor
        
        tsm = torch.bmm(norm_vid, norm_vid.transpose(1, 2))  # [bsz, L_src, L_src]
        
        mask = torch.tensor([
            [1., 1., 0., -1., -1.],
            [1., 1., 0., -1., -1.],
            [0., 0., 0., 0., 0.],
            [-1., -1., 0., 1., 1.],
            [-1., -1., 0., 1., 1.]
        ], device=device)
        mask_size = mask.size(0)
        mask = mask.view(1, 1, mask_size, mask_size)  # [1, 1, 5, 5]
        
        pad_tsm = nn.ZeroPad2d(mask_size // 2)(tsm.unsqueeze(1))  # [bsz, 1, L_src+padding, L_src+padding]
        score = F.conv2d(pad_tsm, mask).squeeze(1)  # [bsz, L_src, L_src]
        score = torch.diagonal(score, dim1=1, dim2=2)  # [bsz, L_src]
        
        tau = score.mean(dim=1, keepdim=True).expand(-1, L_src)  # [bsz, L_src]

        L_vid = torch.sum(src_vid_mask.to(torch.int), dim=1)  # [bsz]
        
        
        for i in range(bsz):
            score[i, 0] = 100  
            score[i, L_vid[i]-1] = 100  
        
        score_r = torch.roll(score, shifts=1, dims=1)
        score_l = torch.roll(score, shifts=-1, dims=1)
        bnds = torch.where((score_r <= score) & (score_l <= score) & (tau <= score), 1., 0.)
        
        for i in range(bsz):
            bnd_indices = torch.nonzero(bnds[i] == 1, as_tuple=False).squeeze(1)
            num_bnds = bnd_indices.size(0)
            
            if num_bnds >= 2:
                
                prev_indices = bnd_indices[:-1]  
                curr_indices = bnd_indices[1:]  
                
                span_widths = curr_indices - prev_indices
                
                valid_mask = span_widths <= L_vid[i] * self.args.span_width_threshold
                
                valid_prev = prev_indices[valid_mask]
                valid_curr = curr_indices[valid_mask]
                valid_spans = torch.stack([valid_prev, valid_curr], dim=1)  # [num_valid, 2]
                
                if valid_spans.size(0) > 0:
                    centers = ((valid_spans[:, 0] + valid_spans[:, 1]) / 2 / L_vid[i].float())
                    widths = ((valid_spans[:, 1] - valid_spans[:, 0]) / L_vid[i].float())
                    
                    num_spans = min(valid_spans.size(0), max_spans)
                    
                    pseudo_event_spans[i, :num_spans, 0] = centers[:num_spans]
                    pseudo_event_spans[i, :num_spans, 1] = widths[:num_spans]
                    pseudo_event_spans_used[i, :num_spans, :] = valid_spans[:num_spans, :]
                    pseudo_event_spans_mask[i, :num_spans] = True
                    continue
            

            pseudo_event_spans[i, 0, 0] = 0.5  
            pseudo_event_spans[i, 0, 1] = 1.0 
            pseudo_event_spans_used[i, 0, 0] = 0 
            pseudo_event_spans_used[i, 0, 1] = L_vid[i].item() - 1 
            pseudo_event_spans_mask[i, 0] = True
        
        return pseudo_event_spans, pseudo_event_spans_used
    
    def adjust_scores_with_event_prior(self, scores, boundaries, event_prior_mask, video_duration):
        """ 
            Optimized function to adjust scores based on event prior mask - purely computational optimization with unchanged results
                    
                    Args: 
            scores: raw confidence scores [num_proposals] 
            boundaries: proposal boundaries [num_proposals, 2] include start and end times 
            event_prior_mask: event a priori mask [batch_size, video_length] or [video_length] 
            video_duration: video duration
                        
                    Returns: 
            adjusted_scores: adjusted scores [num_proposals] 
        """
        if event_prior_mask.dim() > 1:
            video_length = event_prior_mask.shape[1]
            event_prior_mask = event_prior_mask[0] 
        else:
            video_length = event_prior_mask.shape[0]
        
        max_val = event_prior_mask.max()
        if max_val > 0:
            event_prior_mask = event_prior_mask / max_val
        
        scores_min = scores.min()
        scores_max = scores.max()
        scores_range = scores_max - scores_min + 1e-6
        normalized_scores = (scores - scores_min) / scores_range
        
        weight = self.args.score_weight

        
        start_times = torch.clamp(boundaries[:, 0], 0.0, video_duration)
        end_times = torch.clamp(boundaries[:, 1], 0.0, video_duration)

        start_indices = torch.floor(start_times * video_length / video_duration).long()
        end_indices = torch.ceil(end_times * video_length / video_duration).long().clamp(max=video_length-1)
        

        adjusted_scores = torch.zeros_like(scores)
        

        num_proposals = boundaries.shape[0]
        for i in range(num_proposals):
            start_idx = start_indices[i].item()
            end_idx = end_indices[i].item()
            

            if start_idx >= video_length or end_idx < 0 or start_idx >= end_idx:
                adjusted_scores[i] = normalized_scores[i] * 0.01  # 无效proposal使用低权重
                continue
            

            proposal_region = event_prior_mask[start_idx:end_idx+1]
            overlap_score = proposal_region.mean()
            

            smoothed_overlap_score = torch.tanh(overlap_score)
            

            adjusted_scores[i] = normalized_scores[i] * (1 - weight) + smoothed_overlap_score * weight + 1e-6
        
        return adjusted_scores

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, semantic_t_feat,semantic_t_feat_mask,targets=None):
        if vid is not None:
            _count = [v.count('_') for v in vid]
            if self.args.dset_name == 'hl':
                _position_to_cut = [find_nth(v, '_', _count[i]-1) for i, v in enumerate(vid)]
                ori_vid = [v[:_position_to_cut[i]] for i, v in enumerate(vid)]
            else:
                ori_vid = [v for v in vid]
        
        # Project inputs to the same hidden dimension
        src_vid = self.input_vid_proj(src_vid) #[8,742,256]
        
        if self.args.semantic_enhancement:
            src_txt_proj = self.input_txt_proj(src_txt)  # [bsz,8,256]
            semantic_emb = self.input_sematic_txt_proj(semantic_t_feat)  # [bsz,,256]
        
            weight = torch.sigmoid(self.gate)
            src_txt = weight * semantic_emb + (1 - weight) * src_txt_proj 
        else:
            src_txt = self.input_txt_proj(src_txt)  # [bsz,8,256]
        # Add type embeddings
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        # Add position embeddings
        pos_vid = self.position_embed(src_vid, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)

        pseudo_event_spans, pseudo_event_spans_used = self.generate_pseudo_event(src_vid,
                                                        src_vid_mask,
                                                        targets)
        
        event_prior_mask = None
        if pseudo_event_spans_used is not None:

            query_emb = self.pooling(src_txt.float(), src_txt_mask)
            event_features, valid_spans_mask = self.extract_event_features(src_vid, None, pseudo_event_spans_used)

            event_query_sim = self.compute_event_query_similarity(event_features, query_emb,valid_spans_mask)

            event_prior_mask = self.generate_event_prior_mask(
                pseudo_event_spans_used, 
                event_query_sim, 
                src_vid.shape[1]
            )

        txt_dummy = self.dummy_rep_token.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(src_txt.shape[0], 1, 1)
        src_txt_dummy = torch.cat([txt_dummy, src_txt], dim=1)


        mask_txt = torch.tensor([[True] * self.args.num_dummies]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1)
        src_txt_mask_dummy = torch.cat([mask_txt, src_txt_mask], dim=1)

        pos_dummy = self.dummy_rep_pos.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(pos_txt.shape[0], 1, 1)
        pos_txt_dummy = torch.cat([pos_dummy, pos_txt], dim=1)
        src_txt_dummy = src_txt_dummy.permute(1, 0, 2) # (L, batch_size, d)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2) # (L, batch_size, d)

        memory = self.txtproj_encoder(src_txt_dummy, src_key_padding_mask=~(src_txt_mask_dummy.bool()), pos=pos_txt_dummy)
        dummy_token = memory[:self.args.num_dummies].permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)

        src_txt_dummy = torch.cat([dummy_token, src_txt], dim=1)
        mask_txt_dummy = torch.tensor([[True] * self.args.num_dummies]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1)
        src_txt_mask_dummy = torch.cat([mask_txt_dummy, src_txt_mask], dim=1)

        src = torch.cat([src_vid, src_txt_dummy], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask.clone(), src_txt_mask_dummy.clone()], dim=1).bool()        
        pos = torch.cat([pos_vid, pos_txt_dummy], dim=1)

        video_length = src_vid.shape[1]

        video_emb, video_msk, pos_embed, attn_weights, saliency_scores = self.transformer(src, ~mask, pos, video_length=video_length, saliency_proj1=self.saliency_proj1, saliency_proj2=self.saliency_proj2)

        video_emb = video_emb.clone().permute(1, 0, 2)  # (L, batch_size, d) -> (batch_size, L, d)
        video_msk = (~video_msk).int()
        pymid, pymid_msk = self.pyramid(
            video_emb, video_msk, return_mask=self.training == True
        )
        point = self.generator(pymid)

        with torch.autocast("cuda", enabled=False):
            video_emb = video_emb.float()
            query_emb = self.pooling(src_txt.float(), src_txt_mask)
            
            out_class = [self.class_head(e.float()) for e in pymid]
            out_class = torch.cat(out_class, dim=1)
            out_conf = torch.cat(pymid, dim=1)
            out_conf = self.conf_head(out_conf)
            out_class = self.x*out_class+(1-self.x)*out_conf

            if self.coord_head is not None:
                out_coord = [
                    self.coord_head(e.float()).exp() * self.coef[i]
                    for i, e in enumerate(pymid)
                ]
                out_coord = torch.cat(out_coord, dim=1)#[batch_size, num_proposals, 2]，2 表示每个候选片段的时间边界 [start_time, end_time]
            else:
                out_coord = None

            bs, t = src_vid.shape[0], src_vid.shape[1]
            output = dict(_avg_factor=torch.tensor(bs, device=src_txt.device, dtype=torch.float))
            output["saliency_scores"] = saliency_scores
            output["t2vattnvalues"] = (attn_weights[:,:,self.args.num_dummies:] * (src_txt_mask.unsqueeze(1).repeat(1, video_length, 1))).sum(2)
            output["t2vattnvalues"] = torch.clamp(output["t2vattnvalues"], 0, 1)

            if self.training == True:

                output["point"] = point
                output["video_emb"] = video_emb
                output["query_emb"] = query_emb
                output["video_msk"] = video_msk
                output["pymid_msk"] = pymid_msk
                output["out_class"] = out_class #[batch_size, num_proposals, num_classes]
                output["out_coord"] = out_coord 
                
                boundarys = []
                boundarys_true = []
                out_class = out_class.sigmoid()
                video_durations = [qi['duration'] for qi in targets['label']]
                for idx, boundary in enumerate(out_coord):
                    # boundary = boundary.clone()

                    boundary = torch.cat([
                        boundary[:, 0].unsqueeze(1) * -1, #预测的起始时间
                        boundary[:, 1].unsqueeze(1)#预测的结束时间
                    ], dim=1)
                    boundary = boundary * point[:, 3, None].repeat(1, 2)
                    boundary = boundary + point[:, 0, None].repeat(1, 2) #
                    boundary = boundary / (1/self.args.clip_length)
                    # boundary = torch.clamp(boundary, min=0)
                    boundary = torch.cat((boundary, out_class[idx]), dim=-1) #[start_time, end_time, class_scores...]  
                    
                    ###hj__start
                    # 如果启用事件先验筛选，将其用于训练
                    if event_prior_mask is not None and self.args.use_event_prior_in_training:
                        scores = out_class[idx, :, 0]#正类的置信度分数
                        
                        # 使用事件先验对分数进行调整
                        adjusted_scores = self.adjust_scores_with_event_prior(
                            scores, 
                            boundary[:, :2],  # 仅使用边界坐标
                            event_prior_mask[idx] if event_prior_mask.dim() > 1 else event_prior_mask,
                            video_durations[idx]
                        )
                        # 使用调整后的分数重新排序
                        _, inds = adjusted_scores.sort(descending=True)
                    else:
                        # 使用原始分数排序
                        _, inds = out_class[idx, :, 0].sort(descending=True)
                    #hj__end
                    boundary = boundary[inds[:]]
                    boudary_true = torch.clamp(boundary[:,:2], min=0,max=video_durations[idx])
                    boundarys_true.append(boudary_true)
                    boundarys.append(boundary)

                boundarys = torch.stack(boundarys, dim=0)
                boundarys_true = torch.stack(boundarys_true, dim=0)
                output["pred_spans_true"] = boundarys_true
                
                output["pred_spans"] = boundarys

            if self.training == False:
                assert bs == 1, "batch size larger than 1 is not supported for inference"
                out_class = out_class.sigmoid()

                output["_out"] = dict(label=targets.get("label", [None])[0])
                output["_out"]["video_msk"] = video_msk
                output["_out"]["saliency"] = saliency_scores[0]

                if self.coord_head is not None:
                    boundary = out_coord[0]
                    boundary[:, 0] *= -1
                    boundary *= point[:, 3, None].repeat(1, 2)
                    boundary += point[:, 0, None].repeat(1, 2)  
                    # boundary /= 1/self.args.clip_length
                    boundary *= self.args.clip_length
                    boundary = torch.cat((boundary, out_class[0]), dim=-1)  
                    # print(targets)
                    if targets:
                        video_durations = [qi['duration'] for qi in targets['label']]
                    else:
                        video_durations = [150]
                     # ===== 创新点1: 使用事件先验知识过滤proposal =====
                    if event_prior_mask is not None and self.args.use_event_prior_filtering:
                        # 使用 adjust_scores_with_event_prior 函数调整分数
                        confidence = boundary[:, 2]  # 分类置信度分数
                        adjusted_scores = self.adjust_scores_with_event_prior(
                            confidence,  # 原始分数
                            boundary[:, :2],  # 时间边界 [start_time, end_time]
                            event_prior_mask,  # 事件先验掩码
                            video_durations[0]
                        )
                        # 根据调整后的分数排序
                        _, inds = adjusted_scores.sort(descending=True)
                        boundary = boundary[inds[: self.max_num_moment]]
                    else:
                        # 原始分类得分排序方式
                        _, inds = out_class[0, :, 0].sort(descending=True)
                        boundary = boundary[inds[: self.max_num_moment]]

                    output["_out"]["boundary"] = boundary
                    
        if self.training == True and self.args.use_neg:
            ### Neg Pairs ###
            neg_vid = ori_vid[1:] + ori_vid[:1] 
            real_neg_mask = torch.Tensor(element_wise_list_equal(ori_vid, neg_vid)).to(src_txt_dummy.device)
            real_neg_mask = real_neg_mask == False
            if real_neg_mask.sum() != 0:

                src_txt_dummy_neg = torch.cat([src_txt_dummy[1:], src_txt_dummy[0:1]], dim=0)
                src_txt_mask_dummy_neg = torch.cat([src_txt_mask_dummy[1:], src_txt_mask_dummy[0:1]], dim=0)
                src_dummy_neg = torch.cat([src_vid, src_txt_dummy_neg], dim=1)
                mask_dummy_neg = torch.cat([src_vid_mask, src_txt_mask_dummy_neg], dim=1).bool()
                pos_neg = pos.clone() 

                mask_dummy_neg = mask_dummy_neg[real_neg_mask] 
                src_dummy_neg = src_dummy_neg[real_neg_mask] 
                pos_neg = pos_neg[real_neg_mask]
                src_txt_mask_dummy_neg = src_txt_mask_dummy_neg[real_neg_mask]
                
                memory_neg, video_msk, pos_embed, attn_weights_neg, saliency_scores_neg = self.transformer(src_dummy_neg, ~mask_dummy_neg, pos_neg, video_length=video_length, saliency_proj1=self.saliency_proj1, saliency_proj2=self.saliency_proj2)

                output["saliency_scores_neg"] = saliency_scores_neg
                output["src_txt_mask_neg"] = src_txt_mask_dummy_neg

                output["t2vattnvalues_neg"] = (attn_weights_neg[:, :, self.args.num_dummies:] * (src_txt_mask_dummy_neg[:, self.args.num_dummies:].unsqueeze(1).repeat(1, video_length, 1))).sum(2)
                output["t2vattnvalues_neg"] = torch.clamp(output["t2vattnvalues_neg"], 0, 1) 
            else:
                output["saliency_scores_neg"] = None
                output["t2vattnvalues_neg"] = None
            # real_neg_mask = torch.tensor(real_neg_mask, device=src_txt_dummy.device, dtype=torch.bool)
            # # If real_neg_mask is already a tensor, ensure it is the correct type
            # real_neg_mask = real_neg_mask.to(dtype=torch.bool)
            output["real_neg_mask"] = real_neg_mask
            output["dummy_tokens"] = dummy_token
        else:
            # if output["saliency_scores_neg"] is None:
            output["saliency_scores_neg"] = None
            # if output["t2vattnvalues_neg"] is None:
            output["t2vattnvalues_neg"] = None
            # if output["real_neg_mask"] is None:
            output["real_neg_mask"] = None
            # output["real_neg_mask"] = None
            output["dummy_tokens"] = dummy_token

        return output

class SetCriterion(nn.Module):
    """ This class computes the loss."""

    def __init__(self, weight_dict, eos_coef, losses, saliency_margin=1, args=None):
        """ Create the criterion."""
        super().__init__()
        self.args=args
        self.weight_dict = weight_dict
        self.losses = losses
        self.saliency_margin = saliency_margin
        # self.device = args.device
        self.device = 'cuda:1'

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1

        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.l2_criterion = torch.nn.MSELoss().to(self.args.device)
        self.kld_criterion = torch.nn.KLDivLoss(reduction='none').to(self.args.device)
        self.bce_criterion = nn.BCELoss(reduction='none')
        self.SampledNCELoss = SampledNCELoss().to(self.args.device)
        from nncore.nn import build_loss
        self.loss=build_loss(args.cfg.model.loss_cfg)

    def norm(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def loss_labels(self, outputs, targets, log=True):
        sal_score = targets["saliency_all_labels"]
        conf = outputs["out_class"][:, :sal_score.shape[1], 0]

        norm_sal_score = self.norm(sal_score)
        norm_conf = self.norm(conf)
        losses = F.mse_loss(norm_sal_score, norm_conf)
        return {"loss_label": losses}

    def compute_l1_loss(self, pred_spans, relevant_windows):
        """
        计算预测的时间片段边界和真实时间片段边界之间的 L1 损失。

        Args:
            pred_spans (torch.Tensor): 预测的时间片段边界，形状为 [bsz, n, 2]。
            relevant_windows (torch.Tensor): 真实的时间片段边界，形状为 [bsz, m, 2]。

        Returns:
            torch.Tensor: 计算得到的 L1 损失。
        """
        bsz, n, _ = pred_spans.shape
        _, m, _ = relevant_windows.shape
        device = pred_spans.device
        
        # 初始化 L1 损失
        total_l1_loss = torch.tensor(0.0, device=device)

        # 计算每个批次的匹配和损失
        for b in range(bsz):
            # 当前批次的预测和真实片段
            pred = pred_spans[b]  # [n, 2]
            gt = relevant_windows[b]  # [m, 2]
            
            # 如果只有一个真实片段，使用最小距离匹配
            if m == 1:
                # 计算所有预测与唯一真实片段的L1距离
                distances = torch.sum(torch.abs(pred - gt), dim=1)  # [n]
                # 找到最小距离的预测
                min_idx = torch.argmin(distances)
                # 计算最小距离预测的L1损失
                l1_loss = F.l1_loss(pred[min_idx:min_idx+1], gt, reduction='mean')
            else:
                # 对于多个片段，计算距离矩阵
                cost_matrix = torch.cdist(pred, gt, p=1)  # [n, m]
                
                # 使用匈牙利算法找到最佳匹配
                # 转到CPU计算匹配
                cpu_cost_matrix = cost_matrix.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cpu_cost_matrix)
                
                # 转回设备
                row_ind = torch.tensor(row_ind, device=device, dtype=torch.long)
                col_ind = torch.tensor(col_ind, device=device, dtype=torch.long)
                
                # 计算匹配对的L1损失
                l1_loss = F.l1_loss(pred[row_ind], gt[col_ind], reduction='mean')
            
            # 累加batch损失
            total_l1_loss += l1_loss

        # 返回平均 L1 损失
        return total_l1_loss / bsz

    def match_spans(self, pred_spans, relevant_windows):
        """
        优化后的匹配函数，减少不必要的CPU-GPU数据传输，但保持结果不变
        
        Args:
            pred_spans (torch.Tensor): 预测的时间片段边界，形状为 [batch_size, num_proposals, 2]
            relevant_windows (list): 真实的时间片段边界，长度为 batch_size，每个元素是一个二维列表
            
        Returns:
            matched_pred_spans, matched_relevant_windows: 匹配后的预测和真实时间片段
        """
        batch_size, num_proposals, _ = pred_spans.shape
        device = pred_spans.device
        
        # 预分配输出张量列表
        matched_pred_spans = []
        matched_relevant_windows = []

        # 对每个批次分别计算匹配
        for b in range(batch_size):
            if relevant_windows[b] is None or len(relevant_windows[b]) == 0:
                # 如果没有真实标签，使用预测的第一个片段作为匹配
                matched_pred = pred_spans[b, 0:1]  # 取第一个预测
                matched_gt = torch.zeros_like(matched_pred)  # 创建零张量作为伪标签
                
                matched_pred_spans.append(matched_pred)
                matched_relevant_windows.append(matched_gt)
                continue
            # 获取当前批次的预测和真实片段
            pred = pred_spans[b]  # [num_proposals, 2]
            
            # 确保真实片段在正确的设备上
            gt = torch.tensor(relevant_windows[b], dtype=torch.float32, device=device)  # [num_targets, 2]
            
            # 如果只有一个真实片段，可以简化为最小距离匹配
            if gt.size(0) == 1:
                # 计算L1距离
                distances = torch.sum(torch.abs(pred - gt.unsqueeze(0)), dim=1)
                # 找到最小距离的索引
                best_idx = torch.argmin(distances)
                
                matched_pred_spans.append(pred[best_idx:best_idx+1])
                matched_relevant_windows.append(gt)
            else:
                # 对于多个片段，使用匈牙利算法
                # 计算L1距离矩阵
                cost_matrix = torch.cdist(pred, gt, p=1)
                
                # 缓存原始形状以便后续使用
                original_shape = cost_matrix.shape
                
                # 转换为numpy数组进行匈牙利算法计算
                cost_numpy = cost_matrix.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_numpy)
                
                # 将结果转回到GPU
                row_ind = torch.tensor(row_ind, device=device, dtype=torch.long)
                col_ind = torch.tensor(col_ind, device=device, dtype=torch.long)
                
                # 收集匹配的spans
                matched_pred_spans.append(pred[row_ind])
                matched_relevant_windows.append(gt[col_ind])

        # 根据实际匹配数调整输出格式
        max_matches = max(len(spans) for spans in matched_pred_spans)
        
        # 填充张量以便能够堆叠
        for i in range(batch_size):
            curr_len = len(matched_pred_spans[i])
            if curr_len < max_matches:
                # 如果当前匹配少于最大匹配数，填充重复的最后一个匹配
                padding = matched_pred_spans[i][-1:].repeat(max_matches - curr_len, 1)
                matched_pred_spans[i] = torch.cat([matched_pred_spans[i], padding], dim=0)
                
                padding = matched_relevant_windows[i][-1:].repeat(max_matches - curr_len, 1)
                matched_relevant_windows[i] = torch.cat([matched_relevant_windows[i], padding], dim=0)
        
        # 堆叠为张量
        matched_pred_spans = torch.stack(matched_pred_spans, dim=0)
        matched_relevant_windows = torch.stack(matched_relevant_windows, dim=0)
        
        return matched_pred_spans, matched_relevant_windows
    
    def temporal_iou(self, spans1, spans2):
        """
        计算两个时间片段集合之间的 IoU 和并集。

        Args:
            spans1: (bsz, N, 2) torch.Tensor, 每行定义一个时间片段 [start, end]
            spans2: (bsz, M, 2) torch.Tensor, 每行定义一个时间片段 [start, end]

        Returns:
            iou: (bsz, N, M) torch.Tensor, 两个时间片段之间的 IoU
            union: (bsz, N, M) torch.Tensor, 两个时间片段之间的并集长度
        """
        # 计算交集
        inter_start = torch.max(spans1[:, :, None, 0], spans2[:, None, :, 0])  # (bsz, N, M)
        inter_end = torch.min(spans1[:, :, None, 1], spans2[:, None, :, 1])    # (bsz, N, M)
        intersection = (inter_end - inter_start).clamp(min=0)                  # (bsz, N, M)

        # 计算每个时间片段的长度
        len1 = (spans1[:, :, 1] - spans1[:, :, 0]).clamp(min=0)                # (bsz, N)
        len2 = (spans2[:, :, 1] - spans2[:, :, 0]).clamp(min=0)                # (bsz, M)

        # 计算并集
        union = len1[:, :, None] + len2[:, None, :] - intersection             # (bsz, N, M)

        # 计算 IoU
        iou = intersection / union.clamp(min=1e-6)                            # (bsz, N, M)
        return iou, union

    def generalized_temporal_iou(self, spans1, spans2):
        """
        计算 Generalized Temporal IoU。

        Args:
            spans1: (bsz, N, 2) torch.Tensor, 每行定义一个时间片段 [start, end]
            spans2: (bsz, M, 2) torch.Tensor, 每行定义一个时间片段 [start, end]

        Returns:
            giou: (bsz, N, M) torch.Tensor, Generalized IoU
        """
        spans1 = spans1.float()
        spans2 = spans2.float()

        # 确保时间片段的结束时间大于等于开始时间
        assert (spans1[:, :, 1] >= spans1[:, :, 0]).all(), "spans1 的结束时间必须大于等于开始时间"
        assert (spans2[:, :, 1] >= spans2[:, :, 0]).all(), "spans2 的结束时间必须大于等于开始时间"

        # 计算 IoU 和并集
        iou, union = self.temporal_iou(spans1, spans2)  # 调用 temporal_iou 函数

        # 计算包围区域的长度
        enclosing_start = torch.min(spans1[:, :, None, 0], spans2[:, None, :, 0])  # (bsz, N, M)
        enclosing_end = torch.max(spans1[:, :, None, 1], spans2[:, None, :, 1])    # (bsz, N, M)
        enclosing_area = (enclosing_end - enclosing_start).clamp(min=0)            # (bsz, N, M)

        # 计算 Generalized IoU
        giou = iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)     # (bsz, N, M)
        return giou

    def loss_span(self, outputs, targets, log=True):
        """
        计算 L1 损失。
        """
        # 获取预测的时间片段和真实时间片段
        pred_spans = outputs["pred_spans_true"][:,:,:2]
        relevant_windows = [qi["relevant_windows"] for qi in targets["label"]]

        # 匹配预测时间片段和真实时间片段
        matched_pred_spans, matched_relevant_windows = self.match_spans(pred_spans, relevant_windows)

        # 计算 L1 损失
        loss_l1 = self.compute_l1_loss(matched_pred_spans, matched_relevant_windows)
        giou = self.generalized_temporal_iou(matched_pred_spans, matched_relevant_windows)
        # loss_giou = (1-giou).mean() 
        # 计算对角线元素的平均值作为批次的GIoU
        batch_size = matched_pred_spans.size(0)
        min_size = min(matched_pred_spans.size(1), matched_relevant_windows.size(1))
        
        # 获取对角线元素
        batch_indices = torch.arange(batch_size, device=pred_spans.device)
        match_indices = torch.arange(min_size, device=pred_spans.device)
        diagonal_giou = giou[batch_indices[:, None], match_indices[None, :], match_indices[None, :]]
        
        # 计算平均GIoU损失
        loss_giou = (1 - diagonal_giou).mean()
        
        return {"loss_l1": loss_l1, "loss_giou": loss_giou}
    
    def loss_saliency(self, outputs, targets, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        # Neg pair loss
        if outputs["saliency_scores_neg"] is not None: ## When batch size is not 1 (negative pair exists)
            vid_token_mask = outputs["video_msk"]
            real_neg_mask = outputs["real_neg_mask"]
            saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
            loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat([saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

            saliency_scores = outputs["saliency_scores"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair * 0.
            else:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
                
            ########### Saliency loss to t2v attn weights ##############
            """higher scores for positive clips"""
            vid_token_mask = outputs["video_msk"]
            # Neg pair loss

            if outputs["t2vattnvalues_neg"] is not None:
                saliency_scores_neg = outputs["t2vattnvalues_neg"].clone()  # (N, L)
                loss_neg_pair_attn = (- torch.log(1. - saliency_scores_neg) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat(
                [saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (
                        1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive_attn = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive_attn = loss_rank_contrastive_attn + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive_attn / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

            saliency_scores = outputs["t2vattnvalues"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn * 0 + loss_saliency_attn
            else:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn + loss_saliency_attn
            loss_saliency = loss_saliency + (loss_saliency_attn * self.args.lw_wattn)
            
        else: ## when batch size == 1
            vid_token_mask = outputs["video_msk"]
            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)  # no drop
                pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            loss_saliency = loss_saliency + loss_rank_contrastive
            ########### Saliency loss to t2v attn weights ##############
            """higher scores for positive clips"""
            vid_token_mask = outputs["video_msk"]
            saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)  # no drop
                pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive / 12

            saliency_scores = outputs["t2vattnvalues"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
            saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)

            loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_saliency_attn 
            loss_saliency += (loss_saliency_attn * self.args.lw_wattn)
        return {"loss_saliency": loss_saliency}

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "span":self.loss_span,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, **kwargs)

    def extract_relevant_windows(self, data_list):
        all_windows = [instance['relevant_windows'] for instance in data_list]
        max_len = max(len(windows) for windows in all_windows)

        padded_windows = []
        for windows in all_windows:
            new_windows = windows.copy()  
            while len(new_windows) < max_len:
                new_windows.append([float('inf'), float('inf')])
            padded_windows.append(new_windows)
        
        result_tensor = torch.tensor(padded_windows, dtype=torch.float32)
        
        return result_tensor

    def forward(self, batch, outputs, targets):
        """ This performs the loss computation."""
        losses = {}
        new_outputs = {}
        new_outputs["boundary"] = self.extract_relevant_windows(batch[0]).to(self.device) if batch[0][0]['relevant_windows'] != None else None
        new_outputs["saliency"] = targets["saliency_all_labels"]
        new_outputs["pos_clip"] = targets["saliency_pos_labels"][:, 0].unsqueeze(1)
        new_outputs["label"] = batch[0]
        new_outputs["fps"] = targets["fps"]
        new_outputs.update(outputs)

        losses = self.loss(new_outputs, outputs)
        # Compute all the requested losses
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses

class Parameter(nn.Parameter):
    """
    An :obj:`nn.Parameter` class that supports multiple inputs initializes the
    parameters using a scaled normal distribution.
    """

    def __new__(cls, *args, requires_grad=True, **kwargs):
        if torch.is_tensor(args[0]):
            data = args[0]
        elif isinstance(args[0], float):
            data = torch.Tensor([args[0]])
        elif isinstance(args[0], (list, tuple)):
            data = torch.randn(args[0], **kwargs) / args[0][-1]**0.5
        else:
            data = torch.randn(args, **kwargs) / args[-1]**0.5

        return torch.Tensor._make_subclass(cls, data, requires_grad)

class SampledNCELoss(nn.Module):

    def __init__(self,
                 temperature=0.07,
                 max_scale=100,
                 learnable=False,
                 direction=('row', 'col')):
        super(SampledNCELoss, self).__init__()

        scale = torch.Tensor([math.log(1 / temperature)])

        if learnable:
            self.scale = Parameter(scale)
        else:
            self.register_buffer('scale', scale)

        self.temperature = temperature
        self.max_scale = max_scale
        self.learnable = learnable
        self.direction = (direction, ) if isinstance(direction, str) else direction

    def extra_repr(self):
        return ('temperature={}, max_scale={}, learnable={}, direction={}, loss_weight={}'
                .format(self.temperature, self.max_scale, self.learnable, self.direction,
                        self.loss_weight))

    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        batch_inds = torch.arange(video_emb.size(0), device=video_emb.device)

        pos_scores = saliency[batch_inds, pos_clip].unsqueeze(-1)
        loss_msk = (saliency <= pos_scores) * video_msk

        scale = self.scale.exp().clamp(max=self.max_scale)
        i_sim = F.cosine_similarity(video_emb, query_emb, dim=-1) * scale
        i_sim = i_sim + torch.where(loss_msk > 0, .0, float('-inf'))

        loss = 0

        if 'row' in self.direction:
            i_met = F.log_softmax(i_sim, dim=1)[batch_inds, pos_clip]
            loss = loss - i_met.sum() / i_met.size(0)

        if 'col' in self.direction:
            j_sim = i_sim.t()
            j_met = F.log_softmax(j_sim, dim=1)[pos_clip, batch_inds]
            loss = loss - j_met.sum() / j_met.size(0)

        return loss

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model1(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = EventSem(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        input_dropout=args.input_dropout,
        n_input_proj=args.n_input_proj,
        strides=args.cfg.model.strides,
        buffer_size=args.cfg.model.buffer_size,
        max_num_moment=args.cfg.model.max_num_moment,
        pyramid_cfg=args.cfg.model.pyramid_cfg,
        pooling_cfg=args.cfg.model.pooling_cfg,
        coord_head_cfg=args.cfg.model.coord_head_cfg,
        args=args
    )

    weight_dict = {"loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency,
                   'loss_reg': args.lw_reg,
                   "loss_cls": args.lw_cls,
                   "loss_sal": args.lw_sal,
                   "loss_l1":args.lw_l1,
                   "loss_giou":args.lw_giou
                   }

    losses = ["saliency", 'labels','span']

    criterion = SetCriterion(
        weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, saliency_margin=args.saliency_margin, args=args
    )
    criterion.to(device)
    return model, criterion