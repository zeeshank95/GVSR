from pathlib import Path
import os
import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode as CN
from typing import List, Dict
from data.vsitu_vocab.create_vb_voc import vb_vocab
from munch import Munch
import numpy as np
from collections import Counter
from utils.dat_utils import (
    DataWrap,
    get_dataloader,
    simple_collate_dct_list,
    coalesce_dicts,
    arg_mapper,
    pad_words_new,
    pad_tokens,
    read_file_with_assertion,
    load_obj_tsv,
)
from transformers import GPT2TokenizerFast
import pdb

def st_ag(ag):
    return f"<{ag}>"


def end_ag(ag):
    return f"</{ag}>"


def enclose_ag(agname, ag_str):
    return f"{st_ag(agname)} {ag_str} {end_ag(agname)}"


def enclose_ag_st(agname, ag_str):
    return f"{st_ag(agname)} {ag_str}"


class VsituDS(Dataset):
    def __init__(self, cfg: CN, comm: Dict, split_type: str):
        self.full_cfg = cfg
        self.cfg = cfg.ds.vsitu
        self.task_type = self.full_cfg.task_type
        
        self.comm = Munch(comm)
        self.split_type = split_type
        if len(comm) == 0:
            self.set_comm_args()

        assert self.full_cfg.ds.val_set_type == "lb"
        self.full_val = True
        self.read_files(self.split_type)

        self.itemgetter = getattr(self, "grounded_vb_arg_item_getter")
        self.comm.dct_id = "gpt2_hf_tok"


    def set_comm_args(self):

        # self.comm.vb_id_vocab = read_file_with_assertion(
        #     self.cfg.vocab_files.verb_id_vocab, reader="pickle"
        # )

        self.comm.vb_id_vocab = vb_vocab()

        self.comm.gpt2_hf_tok = read_file_with_assertion(
            self.cfg.vocab_files.new_gpt2_vb_arg_vocab, reader="pickle"
        )

        self.comm.gpt2_hf_tok.SPECIAL_TOKENS_ATTRIBUTES.append('start_token')
        special_tokens = {"start_token":"<|startoftext|>"}
        self.comm.gpt2_hf_tok.add_special_tokens(special_tokens)
        id_sos = self.comm.gpt2_hf_tok.encode("<|startoftext|>")[0]
        self.comm.gpt2_hf_tok.start_token_id = id_sos

        self.comm.arg_role_vocab=Arg_Role_Vocab()

        self.comm.path_obj_feats = self.cfg.vsitu_objects_11_frames_dir
        self.comm.vsitu_GT_ann_grounding = self.cfg.vsitu_GT_ann_grounding

        def ptoken_id(self):
            return self.pad_token_id

        def unktoken_id(self):
            return self.unk_token_id

        def eostoken_id(self):
            return self.eos_token_id
        
        def bostoken_id(self):
            return self.bos_token_id


        GPT2TokenizerFast.pad = ptoken_id
        GPT2TokenizerFast.unk = unktoken_id
        GPT2TokenizerFast.eos = eostoken_id
        GPT2TokenizerFast.bos = bostoken_id

        self.comm.num_ev = self.cfg.num_ev
        self.comm.num_frms = self.cfg.num_frms
        self.comm.num_objs_per_frm = self.cfg.num_objs_per_frm
        self.comm.max_num_roles_per_event = self.cfg.max_num_roles_per_event
        self.comm.d_vid_emb = self.cfg.d_vid_emb
        self.comm.d_obj_emb = self.cfg.d_obj_emb
        

    def read_files(self, split_type: str): 
        split_files_cfg = self.cfg.split_files_lb
        vsitu_ann_files_cfg = self.cfg.vsitu_ann_files_lb
        vinfo_files_cfg = self.cfg.vinfo_files_lb

        self.vseg_lst = read_file_with_assertion(split_files_cfg[split_type])
        vseg_ann_lst = read_file_with_assertion(vsitu_ann_files_cfg[split_type])

        vsitu_ann_dct = {}
        for vseg_ann in vseg_ann_lst:
            vseg = vseg_ann["Ev1"]["vid_seg_int"]
            if vseg not in vsitu_ann_dct:
                vsitu_ann_dct[vseg] = []
            vsitu_ann_dct[vseg].append(vseg_ann)
        self.vsitu_ann_dct = vsitu_ann_dct

        if "valid" in split_type or "test" in split_type:
            vseg_info_lst = read_file_with_assertion(vinfo_files_cfg[split_type])
            vsitu_vinfo_dct = {}
            for vseg_info in vseg_info_lst:
                vseg = vseg_info["vid_seg_int"]
                assert vseg not in vsitu_vinfo_dct
                assert len(vseg_info["vbid_lst"]["Ev1"]) >= 9
                vid_seg_ann_lst = [
                    {
                        f"Ev{eix}": {"VerbID": vseg_info["vbid_lst"][f"Ev{eix}"][ix]}
                        for eix in range(1, 6)
                    }
                    for ix in range(len(vseg_info["vbid_lst"]["Ev1"]))
                ]
                vseg_info["vb_id_lst_new"] = vid_seg_ann_lst
                vsitu_vinfo_dct[vseg] = vseg_info
            self.vsitu_vinfo_dct = vsitu_vinfo_dct

    def __len__(self) -> int:
        if self.full_cfg.debug_mode:
            return 30
        return len(self.vseg_lst)

    def __getitem__(self, index: int) -> Dict:
        return self.itemgetter(index)


    def get_vb_data(self, vid_seg_ann_lst: List):
        voc_to_use = self.comm.vb_id_vocab
        label_lst_all_ev = []
        label_lst_mc = []
        for ev in range(1, 6):
            label_lst_one_ev = []
            for vseg_aix, vid_seg_ann in enumerate(vid_seg_ann_lst):
                if vseg_aix == 10:
                    break
                vb_id = vid_seg_ann[f"Ev{ev}"]["VerbID"]

                if vb_id in voc_to_use.indices:
                    label = voc_to_use.indices[vb_id]
                else:
                    label = voc_to_use.unk_index
                label_lst_one_ev.append(label)
            label_lst_all_ev.append(label_lst_one_ev)
            mc = Counter(label_lst_one_ev).most_common(1)
            label_lst_mc.append(mc[0][0])

        label_tensor_large = torch.full((5, 10), voc_to_use.pad_index, dtype=torch.long)
        label_tensor_large[:, : len(vid_seg_ann_lst)] = torch.tensor(label_lst_all_ev)
        label_tensor10 = label_tensor_large
        label_tensor = torch.tensor(label_lst_mc)

        return {"label_tensor10": label_tensor10, "label_tensor": label_tensor}


    def get_vb_arg_SRL_dct(self, vid_seg_ann_lst: List):

        voc_to_use = self.comm.vb_id_vocab
        word_voc = self.comm.gpt2_hf_tok
        arg_role_voc = self.comm.arg_role_vocab

        vb_all_ev = []
        args_all_ev = []
        args_len_all_ev = []
        role_tgt_all_ev = []
        
        arg_SRL_seq_all_ev = []
        arg_SRL_seq_len_all_ev = []
        pad_index = word_voc.pad_token_id
        pad_idx_arg_role = arg_role_voc.pad_idx

        for ev in range(1, 6):
            vb_lst = []
            arg_roles_lst = []
            role_tgt_lst = []
            arg_roles_len_lst = []
            arg_noun_seq_lst = []
            arg_noun_seq_len_lst = []

            for vsix, vid_seg_ann in enumerate(vid_seg_ann_lst):
                ann1 = vid_seg_ann[f"Ev{ev}"]
                vb_id = ann1["VerbID"]
                
                if vb_id in voc_to_use.indices:
                    label = voc_to_use.indices[vb_id]
                else:
                    label = voc_to_use.unk_index
                
                vb_lst.append(label)

                arg_lst = list(ann1["Arg_List"].keys())
                arg_lst_sorted = sorted(arg_lst, key=lambda x: int(ann1["Arg_List"][x])) # [Arg0 (thing fallen), ...]
                arg_str_dct = ann1["Args"] # {Arg0 (thing fallen): 'man in wetsuit', ...}

                role_tgt_lst_temp = [0.]*12
                arg_lst_temp = [pad_idx_arg_role]*6
                arg_lst_len_temp = [0]*6
                arg_noun_lst_temp = [[pad_index]*15]*6
                arg_noun_lst_len_temp = [[0]*15]*6
                
                for i_ag,ag in enumerate(arg_lst_sorted):
                    if i_ag<6:
                        ag_n = arg_mapper(ag)
                        arg_role = arg_role_voc[ag_n]
                        arg_lst_temp[i_ag] = arg_role
                        role_tgt_lst_temp[arg_role] = 1.
                        arg_lst_len_temp[i_ag] = 1

                        arg_str = arg_str_dct[ag] # 'man in wetsuit'
                        seq = arg_str
                        seq_padded, seq_len = pad_words_new(
                            seq,
                            max_len=15,
                            wvoc=word_voc,
                            append_eos=True,
                            use_hf=True,
                            pad_side="right",
                        )
                        seq_padded = seq_padded.tolist()
                        arg_noun_lst_temp[i_ag] = seq_padded
                        arg_noun_lst_len_temp[i_ag] = seq_len

                arg_roles_lst.append(arg_lst_temp)
                role_tgt_lst.append(role_tgt_lst_temp)
                arg_roles_len_lst.append(arg_lst_len_temp)
                arg_noun_seq_lst.append(arg_noun_lst_temp)
                arg_noun_seq_len_lst.append(arg_noun_lst_len_temp)
            
            role_tgt_all_ev.append(role_tgt_lst)
            vb_all_ev.append(vb_lst)
            args_all_ev.append(arg_roles_lst)
            args_len_all_ev.append(arg_roles_len_lst)
            
            arg_SRL_seq_all_ev.append(arg_noun_seq_lst)
            arg_SRL_seq_len_all_ev.append(arg_noun_seq_len_lst)

        vb_arg_noun = {
                    'verbs_all_ev': torch.tensor(vb_all_ev).long(),
                    'args_all_ev': torch.tensor(args_all_ev).long(),
                    'args_len_all_ev':torch.tensor(args_len_all_ev).long(),
                    'arg_noun_seq_all_ev':torch.tensor(arg_SRL_seq_all_ev).long(),
                    'arg_noun_seq_len_all_ev':torch.tensor(arg_SRL_seq_len_all_ev).long(),
                    'role_tgt_all_ev':torch.tensor(role_tgt_all_ev),
                    }

        return vb_arg_noun


    def get_all_bb_11_frames(self, vid_name):
        num_o = self.comm.num_objs_per_frm
        num_f = self.comm.num_frms

        path_vid_obj_feats = os.path.join(self.comm.path_obj_feats, vid_name+'.tsv')
        data = load_obj_tsv(path_vid_obj_feats)
        data = data[:num_f]
        
        feats_11_frames = torch.empty([num_f,num_o,2048], dtype=torch.float32)
        for i,feats in enumerate(data):
            feats_11_frames[i] = torch.from_numpy(feats['features'].copy())

        boxes_11_frames = torch.empty([num_f,num_o,4], dtype=torch.float32)
        for j,feats in enumerate(data):
            boxes_11_frames[j] = torch.from_numpy(feats['boxes'].copy())
        
        img_size_11_frames = torch.empty([11,2], dtype=torch.float32)
        for j,feats in enumerate(data):
            img_size_11_frames[j][0] = feats['img_h']
            img_size_11_frames[j][1] = feats['img_w']

        objects_dict = { "feats_11_frames": feats_11_frames, #11x15x2048
                        "boxes_11_frames": boxes_11_frames, #11x15x4
                        "img_size": img_size_11_frames,
                    }
        return objects_dict

    def get_all_grounding_gt(self, vid_name, vid_seg_ann_lst: List):
        path_grounding_gt = self.comm.vsitu_GT_ann_grounding
        grounding_dict = read_file_with_assertion(path_grounding_gt)
        ann_grounding = grounding_dict[vid_name]
        arg_role_voc = self.comm.arg_role_vocab
        pad_idx_arg_role = arg_role_voc.pad_idx
        gt_frames_all_ev = []
        gt_boxes_all_ev = []
        gt_frames_mask_all_ev = []

        for ev in range(1, 6):
            frames_list = [[pad_idx_arg_role]*11]*6
            boxes_list = [[[pad_idx_arg_role]*4]*11]*6
            frames_mask_list = [[0]*11]*6
            ann_grounding_event = ann_grounding[f"Ev{ev}"]
            ann1 = vid_seg_ann_lst[0][f"Ev{ev}"]
            arg_lst = list(ann1["Arg_List"].keys())
            arg_lst_sorted = sorted(arg_lst, key=lambda x: int(ann1["Arg_List"][x])) # [Arg0 (thing fallen), ...]

            for i_ag,ag in enumerate(arg_lst_sorted):
                if i_ag<6:
                    ag_n = arg_mapper(ag)
                    for key in ann_grounding_event.keys():
                        if ag_n in key:
                            ag_n_full = key
                            break
                    
                    if ann_grounding_event[ag_n_full] != '_':
                        frames = ann_grounding_event[ag_n_full]['Frames']
                        frames_padded = [pad_idx_arg_role]*11
                        bboxes = ann_grounding_event[ag_n_full]['Bboxes']
                        bboxes_padded = [[pad_idx_arg_role]*4]*11

                        if len(frames) > 11:
                            frames = frames[0:11]
                            bboxes = bboxes[0:11]

                        frames_mask_list[i_ag] = [1]*len(frames)+[0]*(11-len(frames))                         
                        frames_padded[0:len(frames)] = frames
                        bboxes_padded[0:len(bboxes)] = bboxes
                        frames_list[i_ag] = frames_padded
                        boxes_list[i_ag] = bboxes_padded
                    else:
                        continue
                     
            gt_frames_all_ev.append(frames_list)
            gt_boxes_all_ev.append(boxes_list)
            gt_frames_mask_all_ev.append(frames_mask_list)

        gr_frame_bboxes_dict = {'gt_frames_all_event': torch.tensor(gt_frames_all_ev).long(),
                    'gt_boxes_all_event': torch.tensor(gt_boxes_all_ev).float(),
                    'gt_frames_all_mask': torch.tensor(gt_frames_mask_all_ev).long(),
                    }
        return gr_frame_bboxes_dict

    def get_frm_feats_all(self, idx: int):
        vid_seg_name = self.vseg_lst[idx]
        vid_seg_feat_file = (
            Path(self.cfg.vsitu_frm_feats_dir) / f"{vid_seg_name}_feats.npy"
        )
        vid_feats = read_file_with_assertion(vid_seg_feat_file, reader="numpy")
        vid_feats = torch.from_numpy(vid_feats).float()
        assert vid_feats.size(0) == 5
        return {"frm_feats": vid_feats}

    def get_vb_label_out_dct(self, idx: int):
        vid_seg_name = self.vseg_lst[idx]
        if self.split_type == "train":
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            vid_seg_ann = vid_seg_ann_[0]
            label_out_dct = self.get_vb_data([vid_seg_ann])
        elif "valid" in self.split_type:
            vid_seg_ann_ = self.vsitu_vinfo_dct[vid_seg_name]["vb_id_lst_new"]
            assert len(vid_seg_ann_) >= 9
            label_out_dct = self.get_vb_data(vid_seg_ann_)
        else:
            raise NotImplementedError

        return label_out_dct


    def grounded_vb_arg_item_getter(self, idx: int):
        feats_out_dct = self.get_frm_feats_all(idx)
        feats_out_dct["vseg_idx"] = torch.tensor(idx)
        label_out_dct = self.get_vb_label_out_dct(idx) #verb GT labels
        vid_seg_name = self.vseg_lst[idx]
        
        if self.split_type == "train":
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            vid_seg_ann = vid_seg_ann_[0]
            verb_arg_SRL_dct = self.get_vb_arg_SRL_dct([vid_seg_ann])

        elif "valid" in self.split_type:
            vid_seg_ann_ = self.vsitu_ann_dct[vid_seg_name]
            assert len(vid_seg_ann_) >= 3
            vid_seg_ann_ = vid_seg_ann_[:3]
            verb_arg_SRL_dct = self.get_vb_arg_SRL_dct(vid_seg_ann_)
            gt_grounding_dict = self.get_all_grounding_gt(vid_seg_name, vid_seg_ann_)

        bounding_boxes_dict = self.get_all_bb_11_frames(vid_seg_name)

        if "valid" in self.split_type:
            out_verb_arg_SRL_dct = coalesce_dicts([feats_out_dct, label_out_dct, verb_arg_SRL_dct, bounding_boxes_dict, gt_grounding_dict])
        else:    
            out_verb_arg_SRL_dct = coalesce_dicts([feats_out_dct, label_out_dct, verb_arg_SRL_dct, bounding_boxes_dict])
        return out_verb_arg_SRL_dct


class Arg_Role_Vocab():
    def __init__(self) -> None:
        self.idx2arg = {}
        self.arg_role_vocab = {"Arg0":0, "Arg1":1, "Arg2":2, "Arg3":3, "Arg4":4, "Arg5":5, "AScn":6, "ADir":7, "APrp":8, "AMnr":9, "ALoc":10, "AGol":11}
        for key,value in self.arg_role_vocab.items():
            self.idx2arg[value] = key
        self.pad_id = '<pad_arg>'
        self.pad_idx = 12
        self.arg_role_vocab[self.pad_id]=self.pad_idx
        
    def __getitem__(self, arg):
        return self.arg_role_vocab[arg]

    def idx2arg(self, idx):
        return self.idx2arg[idx]        
    
    def __len__(self):
        return self.arg_role_vocab.__len__()

        
class BatchCollator:
    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm

    def __call__(self, batch):
        out_dict = simple_collate_dct_list(batch)
        return out_dict


def get_data(cfg):
    DS = VsituDS
    BC = BatchCollator

    train_ds = DS(cfg, {}, split_type="train")
    valid_ds = DS(cfg, train_ds.comm, split_type="valid")
    assert cfg.ds.val_set_type == "lb"
    if cfg.only_test:
        raise NotImplementedError
    else:
        test_ds = None

    batch_collator = BC(cfg, train_ds.comm)
    train_dl = get_dataloader(cfg, train_ds, is_train=True, collate_fn=batch_collator)
    valid_dl = get_dataloader(cfg, valid_ds, is_train=False, collate_fn=batch_collator)

    if cfg.only_test:
        test_dl = get_dataloader(
            cfg, test_ds, is_train=False, collate_fn=batch_collator
        )
    else:
        test_dl = None
    data = DataWrap(
        path=cfg.misc.tmp_path, train_dl=train_dl, valid_dl=valid_dl, test_dl=test_dl
    )
    return data