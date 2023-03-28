import copy
import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor
from einops import repeat
import itertools, math

class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)        

    def forward(self, query, key=None, value=None, mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = query.size()
        if key is None:
            klen = qlen 
        else:
            klen = key.size(1)
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))                                          # (bs, n_heads, qlen, dim_per_head)
        if key is None:
            k = shape(self.k_lin(query))                                      # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(query))                                      # (bs, n_heads, qlen, dim_per_head)
        else:
            k = key
            v = value
            k = shape(self.k_lin(k))                                          # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))                                          # (bs, n_heads, qlen, dim_per_head)

        q = q / math.sqrt(dim_per_head)                                       # (bs, n_heads, qlen, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))                           # (bs, n_heads, qlen, klen)
        mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        scores.masked_fill_(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)
        
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)           # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)
        
        context = torch.matmul(weights, v)                                    # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)                                            # (bs, qlen, dim)
        
        attn_scores=weights.sum(dim=1)/n_heads
        attn_scores_max=torch.argmax(attn_scores, dim=-1)
        return self.out_lin(context), attn_scores_max


class Transformer_VO_RO(nn.Module):

    def __init__(self, cfg, comm, pred_vb, pred_role):
        super().__init__()
        self.cfg=cfg
        self.comm=comm
        
        self.pred_vb = pred_vb
        self.pred_role = pred_role

        self.d_inp_vid_emb = self.comm.d_vid_emb
        self.d_inp_obj_emb = self.comm.d_obj_emb

        self.model_cfg = cfg.transformer_VO_RO  
        self.d_vo_enc = self.model_cfg.d_vo_enc
        self.d_ro_dec = self.model_cfg.d_ro_dec
        self.d_ff = self.model_cfg.dim_feedforward
        nhead = self.model_cfg.nhead
        num_enc_lyrs = self.model_cfg.num_encoder_layers
        num_dec_lyrs = self.model_cfg.num_decoder_layers
        dropout = self.model_cfg.dropout 
        activation= self.model_cfg.activation

        self.num_events = self.comm.num_ev
        self.num_frms = self.comm.num_frms
        self.num_objs_per_frm = self.comm.num_objs_per_frm
        self.max_num_roles_per_event = self.comm.max_num_roles_per_event

        self.num_all_objs = self.num_frms * self.num_objs_per_frm
        self.tot_roles = self.num_events * self.max_num_roles_per_event


        self.vid_feat_project = nn.Sequential(
            *[nn.Linear(self.d_inp_vid_emb, self.d_vo_enc), nn.ReLU(), nn.Linear(self.d_vo_enc, self.d_vo_enc)]
        )
        self.obj_feat_dim_reduction = nn.Sequential(nn.Linear(self.d_inp_obj_emb, self.d_vo_enc),
                                             nn.ReLU(),
                                             nn.Linear(self.d_vo_enc, self.d_vo_enc))

        # Spatial 2d position embeding for objects;
        self.obj_spatial_pos_emd = nn.Linear(4, self.d_vo_enc)
        # Position embeding for video 1-5;
        self.event_pos_embed = nn.Embedding(self.num_events, self.d_vo_enc)
        # Object frame level position embedding (0-10)
        self.frame_pos_embed = nn.Embedding(self.num_frms,self.d_vo_enc)
        # video vs object type embedding
        self.input_type_embed = nn.Embedding(2, self.d_vo_enc)
        
        # vid embed and obj embed norm 
        self.vid_enc_emb_norm = nn.LayerNorm(self.d_vo_enc)
        self.obj_enc_emb_norm = nn.LayerNorm(self.d_vo_enc)
        
        self.num_verb_classes = len(comm.vb_id_vocab)
        self.num_role_classes = len(comm.arg_role_vocab)
        self.arg_role_voc = comm.arg_role_vocab

        # Video Object Transformer encoder
        vo_encoder_layer = TransformerEncoderLayer(self.d_vo_enc, nhead, self.d_ff, dropout, activation)
        vo_encoder_norm = nn.LayerNorm(self.d_vo_enc)
        self.vo_encoder = TransformerEncoder(vo_encoder_layer, num_enc_lyrs, vo_encoder_norm)
        
        # classifer (for verb prediction)
        if self.pred_vb:
            self.verb_classifier = nn.Sequential(nn.Linear(self.d_vo_enc, self.d_vo_enc*2),
                                                nn.ReLU(),
                                                nn.Linear(self.d_vo_enc*2, self.num_verb_classes))
            self.criterion_verb = nn.CrossEntropyLoss()
    
        # role Query embedding, total 11 roles
        self.role_embed = nn.Embedding(self.num_role_classes, self.d_ro_dec)

        if self.pred_role:
            self.role_classifer_l1 = nn.Linear(self.d_vo_enc, self.d_vo_enc)
            self.role_classifer_l2 = nn.Linear(self.d_vo_enc, self.num_role_classes-1)

        # RO Decoder
        ro_dec_layer = TransformerDecoderLayer(self.d_ro_dec, nhead, self.d_ff, dropout, activation)
        ro_dec_norm = nn.LayerNorm(self.d_ro_dec)
        self.ro_decoder = TransformerDecoder(ro_dec_layer, num_dec_lyrs, ro_dec_norm) 
       

    def process_vid_obj_inputs(self, inp):
        B, N, D = inp['frm_feats'].size()
        vid_feats = inp['frm_feats'] # Bx5x2304
        vid_emb = self.vid_feat_project(vid_feats) # Bx5xD 
        
        #vid event positions
        positions = vid_emb.new(self.num_events).long() 
        positions = torch.arange(self.num_events, out=positions) 
        vid_pos_emb = repeat(self.event_pos_embed(positions), 'n d -> b n d', b = B) # Bx5xD   

        type_vid = vid_emb.new(self.num_events).long() 
        type_vid =  torch.zeros(self.num_events, out=type_vid) 
        type_vid_emb = repeat(self.input_type_embed(type_vid), 'n d -> b n d', b = B) # Bx5xD   

        vid_feats_5 = vid_emb+vid_pos_emb+type_vid_emb

        # 11 frames, each frame consist of 15 objects
        B,F,N,D = inp['feats_11_frames'].size()
        obj_feats_embd = inp['feats_11_frames'] # Bx11x15x2048
        obj_feats_embd = obj_feats_embd.view(B,F*N,D) # Bx165x2048
        obj_feats_embd = self.obj_feat_dim_reduction(obj_feats_embd) # Bx165x512
        #box cordinates
        obj_bb_pos = inp['boxes_11_frames'].clone() # Bx11x15x4
        #normalise box cordinates to 0-1
        img_size_batch = inp['img_size'] #Bx11x2
        for b_s, vid in enumerate(img_size_batch):
            img_h = vid[0][0]
            img_w = vid[0][1]
            obj_bb_pos[b_s,:,:,0] = obj_bb_pos[b_s,:,:,0]/img_w
            obj_bb_pos[b_s,:,:,1] = obj_bb_pos[b_s,:,:,1]/img_h
            obj_bb_pos[b_s,:,:,2] = obj_bb_pos[b_s,:,:,2]/img_w
            obj_bb_pos[b_s,:,:,3] = obj_bb_pos[b_s,:,:,3]/img_h 

        # Object 2D box spatial position embedding
        obj_bb_pos = obj_bb_pos.view(B,F*N,4) # Bx165x4
        obj_spat_pos_embd = self.obj_spatial_pos_emd(obj_bb_pos) # Bx165x512
        B,F_N,D = obj_spat_pos_embd.size()        
        
        # Add video event position embeddings to objects. 
        # Obejcts from the 11 frames are divided into 5 events [0-4], with objects in 
        # boundary frames corresponding to both the neighboring events.
        pos_level1 = [0,0,0,1,1,2,2,3,3,4,4] 
        event_pos_per_frame_l1 = obj_spat_pos_embd.new_tensor(pos_level1).long()
        frame_event_pos_emb_level1 = repeat(self.event_pos_embed(event_pos_per_frame_l1), 'n d -> b n d', b = B) # Bx11xD
        pos_level2 = [1,2,3,4]
        pos_level2_idx = [2,4,6,8]
        event_pos_per_frame_l2 = obj_spat_pos_embd.new_tensor(pos_level2).long()
        frame_event_pos_emb_level2 = repeat(self.event_pos_embed(event_pos_per_frame_l2), 'n d -> b n d', b = B) # Bx4xD
        frame_event_pos_emb_level1[:,pos_level2_idx] = frame_event_pos_emb_level1[:,pos_level2_idx] + frame_event_pos_emb_level2
        #repeat the frame level event embeddings to object level event embeddings
        obj_event_pos_emb = repeat(frame_event_pos_emb_level1, 'b n d -> b n o d', o = self.num_objs_per_frm)
        obj_event_pos_emb = obj_event_pos_emb.view(B,F_N,D)

        # 11 frame level position embeddings for 11 frames
        frame_pos = vid_emb.new(self.num_frms).long()  
        frame_pos = torch.arange(self.num_frms, out=frame_pos)
        obj_frame_pos_embed = repeat(self.frame_pos_embed(frame_pos), 'n d -> b n o d', b = B, o = self.num_objs_per_frm) # Bx11x15xD
        obj_frame_pos_embed = obj_frame_pos_embed.view(B,self.num_all_objs,D) # Bx165x512     

        #object type embedding
        type_obj = vid_emb.new(self.num_all_objs).long() 
        type_obj =  torch.ones(self.num_all_objs, out=type_obj) 
        type_obj_emb = repeat(self.input_type_embed(type_obj), 'n d -> b n d', b = B) # Bx165xD   

        obj_feats_165 = obj_spat_pos_embd + obj_event_pos_emb + type_obj_emb + obj_frame_pos_embed + obj_feats_embd
        encoder_src = torch.cat([vid_feats_5, obj_feats_165], dim=1)

        #mask for vid_obj encoder
        len_obj_ev = self.num_events+self.num_all_objs
        alen = obj_feats_embd.new(len_obj_ev).long() 
        alen = torch.ones(len_obj_ev, out=alen) 
        attn_mask = alen[None,:].repeat(B,1)
        return encoder_src, attn_mask
    

    def compute_vb_loss(self, inp, vb_pred):
        labels_c1 = (inp["label_tensor"])
        labels_c1 = labels_c1.view(-1)
        vb_pred = vb_pred.view(-1, self.num_verb_classes)
        loss_vb = self.criterion_verb(vb_pred, labels_c1)
        return loss_vb

    def classify_role(self, vid_feats_5):
        pred_roles = self.role_classifer_l1(vid_feats_5) #BxNxD
        pred_roles = F.sigmoid(self.role_classifer_l2(pred_roles)) #BxNx12
        return pred_roles

    def compute_role_loss_enc(self, inp, pred_roles):
        role_tgt = inp['role_tgt_all_ev'][:,:,[0],:].clone()
        B,N,N_,R = role_tgt.size()
        role_tgt = role_tgt.squeeze().view(B*N, R)
        pred_roles = pred_roles.view(B*N, R)
        role_loss = F.binary_cross_entropy(pred_roles, role_tgt)
        return role_loss

    def get_event_aware_attention_map(self, inp):
        atten_mask = inp['feats_11_frames'].new(self.tot_roles, self.num_all_objs).long() 
        B,F,N,D = inp['feats_11_frames'].size()
        atten_mask[:,:] = 0

        num_objs_per_event = self.num_objs_per_frm * 3  
        object_stride = self.num_objs_per_frm * 2 #30
        # objects [0-45] belongs to event 1
        # objects [30-75] belongs to event 2
        #..
        # objects [120-165] belongs to event 5        
        left = -object_stride
        for i in range(self.tot_roles):
            if i % self.max_num_roles_per_event == 0:
                left+=object_stride
                right=left+num_objs_per_event
            atten_mask[i,left:right] = 1
        atten_mask = atten_mask[None,:,:].repeat(B,1,1)
        return atten_mask

    def process_role_query(self, inp, vid_5_feats, pred_roles, inference=False):
        
        if self.pred_role and inference: #Create role query based on the predicted roles during inference only.
            B,N,D = vid_5_feats.size()
            pad_arg = self.arg_role_voc.pad_idx
            selected_roles = vid_5_feats.new(B,N,self.max_num_roles_per_event).long().fill_(pad_arg)
            selected_roles_pad_in_mask = vid_5_feats.new(B,N,self.max_num_roles_per_event).long().fill_(0)

            for sample_idx, sample in enumerate(pred_roles):
                for vid_idx, vid in enumerate(sample):
                    count_role=0
                    for role_idx, role_prob in enumerate(vid): # find all the predicted roles out of 11
                        if role_prob>0.5 and count_role<self.max_num_roles_per_event: #select the roles with prob>0.5
                            selected_roles[sample_idx, vid_idx, count_role]=role_idx
                            selected_roles_pad_in_mask[sample_idx, vid_idx, count_role] = 1
                            count_role+=1
            
            selected_roles_pad_in_mask = selected_roles_pad_in_mask.squeeze().view(B,-1) #Bx30 
            selected_roles = selected_roles.squeeze().view(B,-1) #Bx30 
            selected_roles_embdedding = self.role_embed(selected_roles) #Bx30xD 
        
        else: # Use ground truth roles to model role queries and teacher force SRL.
            B, N, N_, R = inp['args_all_ev'].size() #R=6 for max 6 roles per verb. 
            selected_roles = inp['args_all_ev'][:,:,[0],:].clone() #Bx5x1x6
            selected_roles_pad_mask = inp['args_len_all_ev'][:,:,[0],:].clone() #Bx5x1x6
            selected_roles_pad_in_mask = selected_roles_pad_mask.squeeze().view(B,-1) #Bx30             
            selected_roles = selected_roles.squeeze().view(B,-1) #Bx30 
            selected_roles_embdedding = self.role_embed(selected_roles) #Bx30xD 
            
        vid_embed = vid_5_feats.unsqueeze(2).repeat(1,1,6,1) #Bx5x6xD
        B,N,R,D = vid_embed.size()
        vid_embed = vid_embed.view(B, -1, D) # Bx30xD

        positions = selected_roles.new(self.num_events).long() 
        positions = torch.arange(self.num_events, out=positions) # 5xD  
        positions_emb = repeat(self.event_pos_embed(positions), 'n d -> b n r d', b = B, r = 6) # Bx5x6xD    
        positions_emb = positions_emb.view(B,-1, D) # Bx30xD    
        vid_role_src_query = vid_embed + selected_roles_embdedding + positions_emb

        return vid_role_src_query, selected_roles_pad_in_mask, selected_roles

    
    def forward(self, inp, cfg, inference=False):
        vid_obj_enc_emb, attn_mask_vid_obj = self.process_vid_obj_inputs(inp)

        # Forward VO Transformer Encoder
        vid_obj_out = self.vo_encoder(cfg, vid_obj_enc_emb, src_key_padding_mask=attn_mask_vid_obj)
        vid_enc_out = vid_obj_out[:, 0:self.num_events, :]
        obj_enc_out = vid_obj_out[:, self.num_events:, :]

        vb_pred = self.verb_classifier(vid_enc_out) #Bx5xnum_verb_class

        if self.pred_vb and not inference:
            vb_loss = self.compute_vb_loss(inp, vb_pred)
        else:
            vb_loss = None
        
        if self.pred_role and not inference:
            pred_roles = self.classify_role(vid_enc_out)
            role_loss = self.compute_role_loss_enc(inp, pred_roles)
        else:
            pred_roles = None
            role_loss = None

        # Process the role query for role decoder
        vid_role_src_query, roles_pad_mask_in, selected_roles = self.process_role_query(inp, vid_enc_out, pred_roles, inference=inference)
        roles_pad_mask_in = roles_pad_mask_in.bool()

        # Get event aware cross attention(momory) mask for RO decoder
        attn_mask_obj_event_aware = self.get_event_aware_attention_map(inp)
        obj_role_tgt = torch.zeros_like(vid_role_src_query)

        # Forward RO Transformer Decoder
        grounded_nouns_emb, bb_attn = self.ro_decoder(cfg, tgt=obj_role_tgt, memory=obj_enc_out, tgt_key_padding_mask=roles_pad_mask_in,
                           memory_key_padding_mask=attn_mask_obj_event_aware, query_pos=vid_role_src_query)   

        return vb_pred, vb_loss, pred_roles, role_loss, grounded_nouns_emb, bb_attn, selected_roles


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, cfg, src, src_key_padding_mask: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(cfg, output, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(nhead, d_model, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, cfg, src, src_key_padding_mask: Optional[Tensor] = None):
        src2 = self.norm1(src)
        src2, src_attn_out = self.self_attn(query=src2, mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.activation(self.linear1(src2)))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, cfg, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt         
        for i,layer in enumerate(self.layers):
            if i==0:
                skip_self_attn=True  
            else:
                skip_self_attn=False
            output, bb_attn = layer(cfg, output, memory,
                           tgt_key_padding_mask, memory_key_padding_mask,
                           query_pos, skip_self_attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, bb_attn

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(nhead, d_model, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(nhead, d_model, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, cfg, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None, skip_self_attn=False):

        if skip_self_attn:
            tgt2=tgt
        else:
            tgt2 = self.norm1(tgt)
            tgt2 = tgt2*tgt_key_padding_mask.unsqueeze(-1)            
            k=v=tgt2
            q = tgt2+query_pos
            tgt2, out_self_Atn = self.self_attn(query=q, key=k, value=v, mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)
            
        q_cross = tgt2 + query_pos
        k_cross = memory
        v_cross = memory
        tgt2, cross_attn_weights = self.multihead_attn(q_cross, k_cross, v_cross, mask=memory_key_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))

        tgt = tgt + self.dropout3(tgt2)
        tgt = tgt*tgt_key_padding_mask.unsqueeze(-1)

        return tgt, cross_attn_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    return m