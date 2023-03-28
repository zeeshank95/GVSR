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

    def forward(self, input, mask, kv=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen
        else:
            klen = kv.size(1)
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

        q = shape(self.q_lin(input))                                          # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))                                      # (bs, n_heads, qlen, dim_per_head)
        else:
            k = v = kv
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
        return self.out_lin(context)

class Transformer_Captioning(nn.Module):

    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg=cfg
        self.comm=comm
        self.model_cfg = cfg.transformer_caption  
        self.d_cap_dec = self.model_cfg.d_cap_dec
        dim_feedforward = self.model_cfg.d_feedforward
        self.nhead = self.model_cfg.nhead
        num_caption_decoder_layers = self.model_cfg.num_caption_decoder_layers
        self.max_cap_len = self.model_cfg.max_cap_len
        dropout = self.model_cfg.dropout 
        activation= self.model_cfg.activation
        SRL_voc = len(self.comm.gpt2_hf_tok)
        
        # SRL caption vocab
        self.vocab_embed = nn.Embedding(SRL_voc, self.d_cap_dec)

        # position embeding for SRL caption with a max length of 15
        self.pos_cap_embed = nn.Embedding(self.max_cap_len, self.d_cap_dec)
        
        # SRL caption decoder
        caption_decoder_layer = TransformerDecoderLayer(self.cfg, self.d_cap_dec, self.nhead, dim_feedforward,
                                                dropout, activation)
        caption_decoder_norm = nn.LayerNorm(self.d_cap_dec)
        
        self.caption_decoder = TransformerDecoder(caption_decoder_layer, num_caption_decoder_layers, caption_decoder_norm)                      
                                             
        self.noun_classifier = nn.Linear(self.d_cap_dec, SRL_voc, bias=False)


    def process_SRL_captions(self, inp, grounded_nouns):
        
        #SRL captions for all the roles for all the events in a video
        #5 events, max 6 roles per event, max 15 caption length per role
        noun_seq_all = inp['arg_noun_seq_all_ev'][:,:,[0],:,:]
        B,N,N_,R,L = noun_seq_all.shape #Bx5x1x6x15
        noun_seq_all = noun_seq_all.squeeze() #Bx5x6x15
        noun_seq_all = noun_seq_all.view(B*N*R,L) #B*30x15

        # Caption padding mask.
        noun_seq_all_len = inp['arg_noun_seq_len_all_ev'][:,:,[0],:,:]
        noun_seq_all_len = noun_seq_all_len.squeeze() #Bx5x6x15
        noun_seq_all_len = noun_seq_all_len.view(B*N*R,L) #B*30x15

        # Role padding mask
        selected_roles_pad_mask = inp['args_len_all_ev'][:,:,[0],:] #Bx5x1x6
        B,N,N_,R = selected_roles_pad_mask.shape
        selected_roles_pad_mask_out = selected_roles_pad_mask.clone().squeeze().view(B*N*R).bool() #B*30 

        # remove the empty roles from the noun_sequence tensor
        # batch and role dimensions collapsed (B,30) -> (B*30)  
        # then remove the empty roles, this gives new_num_nouns -> (N) in the first dimension 
        # This allows parallel processing for caption generation
        selected_noun_seq_all = noun_seq_all[selected_roles_pad_mask_out] #New_num_nouns x 15
        selected_noun_seq_all_len = noun_seq_all_len[selected_roles_pad_mask_out] #New_num_nouns x 15

        # Similarly remove the empty grounded_nouns 
        # This allows parallel processing for caption generation
        B,nr,D = grounded_nouns.shape
        grounded_nouns = grounded_nouns.view(B*nr, D)
        grounded_nouns = grounded_nouns[selected_roles_pad_mask_out] #New_num_nouns x D

        # exclude EOS from pred caption
        pred_mask = selected_noun_seq_all_len.clone()
        for row in pred_mask:
            idx = row.sum()-1
            row[idx]=0
        
        # exclude BOS from tgt caption
        tgt_gt = selected_noun_seq_all.clone()
        tgt_mask = selected_noun_seq_all_len.clone()
        tgt_mask[:,0]=0

        return selected_noun_seq_all, selected_noun_seq_all_len, grounded_nouns, pred_mask, tgt_mask, tgt_gt


    def get_causal_mask(self, slen, bs, ref):
        alen = ref.new(slen).long() 
        alen = torch.arange(slen, out=alen) 
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
        return attn_mask    


    def get_context_mask(self, bs, ref):
        alen = ref.new(bs).long() 
        alen = torch.ones(bs, out=alen) 
        alen = alen.unsqueeze(-1)
        return alen    

    def get_attn_gen_mask(self, slen, bs, ref):
        alen = ref.new(slen).long() 
        alen = torch.ones([slen,slen], out=alen)
        att_mask = alen[None, :, :].repeat(bs, 1, 1)
        return att_mask

    def compute_cap_loss(self, cfg, pred, labels):

        loss_cap = F.cross_entropy(pred, labels)
        return loss_cap      


    def forward(self, inp, cfg, grounded_SRL):
        nouns_seq, nouns_seq_len, grounded_nouns_new, pred_mask, tgt_mask, tgt_gt  = self.process_SRL_captions(inp, grounded_SRL)
        
        # query for the transformer caption decoder, teacher forcing applied  
        nouns_seq_embed = self.vocab_embed(nouns_seq) #Nx15xD
        N,L,D = nouns_seq_embed.shape

        positions = nouns_seq_embed.new(L).long() 
        positions = torch.arange(L, out=positions) 
        positions_emb = repeat(self.pos_cap_embed(positions), 'l d -> n l d', n = N) # Nx15xD   

        # Context/memory for the transformer caption decoder 
        grounded_nouns_new=grounded_nouns_new.unsqueeze(1) #Nx1xD

        causal_mask = self.get_causal_mask(L, N, grounded_nouns_new)
        memory_key_padding_mask = self.get_context_mask(N, grounded_nouns_new)
        tgt_key_pad_mask = nouns_seq_len.bool()

        noun_captions = self.caption_decoder(cfg, nouns_seq_embed, grounded_nouns_new,  
                        causal_mask=causal_mask, tgt_key_padding_mask=tgt_key_pad_mask, 
                        memory_key_padding_mask=memory_key_padding_mask, pos=positions_emb) #Nx15x512

        pred_mask = pred_mask.bool()
        tgt_mask = tgt_mask.bool()

        pred_nouns = noun_captions[pred_mask].view(-1, self.d_cap_dec) # all_nouns x D
        pred_nouns = self.noun_classifier(pred_nouns) # all_nouns x noun_vocab

        tgt_nouns_gt = tgt_gt[tgt_mask] #all_nouns
        cap_loss = self.compute_cap_loss(cfg, pred_nouns, tgt_nouns_gt)

        return noun_captions, cap_loss # all_nouns x noun_voacab, all_nouns


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, cfg, tgt, memory,
                causal_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        output = tgt

        for layer in self.layers:
            output = layer(cfg, output, memory, causal_mask=causal_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):

    def __init__(self, cfg, d_model, nhead, dim_feedforward=2048, dropout=0.15, activation="relu"):
        super().__init__()
        self.cfg=cfg
        self.self_attn = MultiHeadAttention(nhead, d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(nhead, d_model, dropout=dropout)

        # Implementation of Feedforward model
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
                causal_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        tgt = tgt+pos
        tgt2 = self.norm1(tgt)
        if tgt_key_padding_mask != None: 
            tgt2 = tgt2*tgt_key_padding_mask.unsqueeze(-1)

        tgt2 = self.self_attn(tgt2, mask=causal_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        query_cross = tgt2
        kv_cross = memory
        tgt2 = self.cross_attn(query_cross, mask=memory_key_padding_mask, kv=kv_cross)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        if tgt_key_padding_mask != None: 
            tgt = tgt*tgt_key_padding_mask.unsqueeze(-1)

        return tgt

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