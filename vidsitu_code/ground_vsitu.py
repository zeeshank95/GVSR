import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_grounded_vsitu import Transformer_VO_RO
from transformer_noun_caption import Transformer_Captioning
from einops import repeat

class LossLambda(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.loss_keys = self.cfg.loss_keys

    def forward(self, mdl_out):
        loss_dict = {}
        for key in self.loss_keys:
            loss_dict[key] = mdl_out[key]
        return loss_dict

class Grounded_VidSitu(nn.Module):
    def __init__(self, cfg, comm) -> None:
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.task_type = self.cfg.task_type 
        if self.task_type == "grounded_srl_GT_vbrole":
            pred_vb = False
            pred_role = False
        elif self.task_type == "grounded_vb_srl_GT_role":
            pred_vb = True
            pred_role = False
        elif self.task_type == "grounded_end-to-end":
            pred_vb = True
            pred_role = True
        self.tx_vo_ro = Transformer_VO_RO(self.cfg, self.comm, pred_vb, pred_role)
        self.tx_cap = Transformer_Captioning(self.cfg, self.comm)
        self.word_voc = self.comm.gpt2_hf_tok

    def forward(self, inp): 
        vb_pred, vb_loss, pred_roles, role_loss, grounded_nouns, bb_attn, selected_roles = self.tx_vo_ro(inp, self.cfg)
        noun_caps, cap_loss = self.tx_cap(inp, self.cfg, grounded_nouns)
        out_dct = {"loss_vb": vb_loss, "loss_SRL": cap_loss, "loss_role": role_loss}
        return out_dct

    def forward_gen(self, inp):
        inp['args_all_ev'] = inp['args_all_ev'][:,:,[0],:]
        inp['verbs_all_ev'] = inp['verbs_all_ev'][:,:,[0]]
        inp['args_len_all_ev'] = inp['args_len_all_ev'][:,:,[0],:]
        vb_preds, vb_loss, pred_roles, role_loss, grounded_nouns, bb_attn, selected_roles = self.tx_vo_ro(inp, self.cfg, inference=True)

        # compute the input and masks for caption decoder
        selected_roles_pad_mask = inp['args_len_all_ev'] #Bx5x1x6
        B,N,N_,R = selected_roles_pad_mask.shape
        selected_roles_pad_mask_out = selected_roles_pad_mask.clone().squeeze().view(B*N*R).bool() #B*30 
        B,nr,D = grounded_nouns.shape
        grounded_nouns = grounded_nouns.view(B*nr, D)
        grounded_nouns = grounded_nouns[selected_roles_pad_mask_out] #New_nouns x 512
        grounded_nouns_new=grounded_nouns.unsqueeze(1) #Nx1x512
        
        start_symbol = self.word_voc.start_token_id
        N,l_,D = grounded_nouns_new.size()
        curr_sent = grounded_nouns_new.new(1).long() 
        curr_sent[0] = start_symbol 
        curr_sent_emb = repeat(self.tx_cap.vocab_embed(curr_sent), 'l d -> n l d', n = N) # Nx1xD   
        
        final_sent_toks = grounded_nouns_new.new().long()
        memory_attn = self.tx_cap.get_context_mask(N, grounded_nouns_new)
        
        max_cap_len = self.cfg.transformer_caption.max_cap_len
        #Autoregressive decoding
        for i in range(max_cap_len):
            p = i+1
            positions = grounded_nouns_new.new(p).long() 
            positions = torch.arange(p, out=positions) 
            positions_emb = repeat(self.tx_cap.pos_cap_embed(positions), 'l d -> n l d', n = N) # NxpxD   
            causal_mask = self.tx_cap.get_attn_gen_mask(p, N, grounded_nouns_new)

            #feed the inputs to the noun caption decoder
            out = self.tx_cap.caption_decoder(self.cfg, curr_sent_emb, 
                            grounded_nouns_new, causal_mask=causal_mask, memory_key_padding_mask=memory_attn,
                            pos=positions_emb)
            
            out_curr = out[:,[-1],:]

            pred_class = self.tx_cap.noun_classifier(out_curr) #N x 1 x vocab
            pred_prob = F.softmax(pred_class, dim=-1)
            
            _, next_words_ids = torch.max(pred_prob, dim = -1) #Nx1
            
            next_words_embd = self.tx_cap.vocab_embed(next_words_ids.clone()) #Nx1xD

            curr_sent_emb = torch.cat([curr_sent_emb, next_words_embd], dim=1) #N x p+1 x D

            final_sent_toks = torch.cat([final_sent_toks, next_words_ids], dim=-1) #N x p

        return vb_preds, final_sent_toks, inp['args_all_ev'].clone().squeeze(), bb_attn #final_sent_toks = Nx15
