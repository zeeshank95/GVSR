"""
Evalution for Vsitu
"""
import os
import torch
from torch import nn
from torch.nn import functional as F
import cv2
import pickle
from pathlib import Path
from utils.trn_utils import (
    progress_bar,
    move_to,
    synchronize,
    is_main_process,
    compute_avg_dict,
    get_world_size,
)
from utils.dat_utils import read_file_with_assertion
from vidsitu_code.evl_fns import EvlFn_Vb, EvalFnCap
import matplotlib.pyplot as plt

class Eval_grvidsitu(nn.Module):
    def __init__(self, cfg, comm, device):
        super().__init__()
        self.cfg = cfg
        self.full_cfg = cfg
        self.comm = comm
        self.device = device
        self.eval_type = self.full_cfg.eval_type 
        self.compute_loss = True

        if self.eval_type == 'SRL_eval':
            self.met_keys = [ "cider", "rouge", "lea", "MacroVb_cider", "MacroArg_cider"]
            
        elif self.eval_type == 'Vb_SRL_eval':
            self.met_keys = ["cider", "rouge", "lea", "MacroVb_cider", "MacroArg_cider", "Per_Ev_Top_1", "Per_Ev_Top_5", "recall_macro_1_th_9", "IoU", "IoU_30", "IoU_50"]

        self.eval_vb = EvalB(cfg, comm)
        self.eval_SRL = EvalB_Gen(cfg, comm)
        
    def dump_groundings(self, grounding_results_list, pred_path):
        pred_path = os.path.join(pred_path, "Visual_grounding")    
        for video_dict in grounding_results_list:  # vid grounding is a list of video level groundings, every video is a dictionary
            video_id = video_dict['vid_id']
            vid_out_dir = os.path.join(pred_path, video_id)
            os.makedirs(vid_out_dir, exist_ok=True)                

            len_frames = len(video_dict['frame_list'])
            for idx in range(len_frames):
                frame_path_in = video_dict['frame_list'][idx]
                caption = video_dict['caption_list'][idx]
                arg_name = caption.split(':')[0]
                frame = cv2.imread(frame_path_in)

                frame_name = arg_name+frame_path_in.split('/')[-1]
                frame_out_pth = os.path.join(vid_out_dir, frame_name)
                coords = video_dict['box_coord_list'][idx]
                
                top_left = (coords[0], coords[1])
                bottom_right = (coords[2], coords[3])
                frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0))
                cv2.putText(frame, str(caption), (coords[0], coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.imwrite(frame_out_pth, frame)
        return

    def forward(self, model, loss_fn, dl, dl_name, rank=0, pred_path=None, mb=None):

        fname_vb = Path(pred_path) / f"{dl_name}_{rank}_vb.pkl"
        fname_SRL = Path(pred_path) / f"{dl_name}_{rank}_srl.pkl"

        model.eval()
        model.to(self.device)
        loss_keys = loss_fn.loss_keys
        val_losses = {k: [] for k in loss_keys}
        nums = []
        results_vb = []
        results_SRL = []
        grounding_result = []

        for batch in progress_bar(dl, parent=mb):
            batch = move_to(batch, self.device)
            b = next(iter(batch.keys()))
            nums.append(batch[b].size(0))
            torch.cuda.empty_cache()

            if self.compute_loss:
                with torch.no_grad():
                    # forward pass to get the losses on validation set
                    out = model(batch)
                    out_loss = loss_fn(out)
                for k in out_loss:
                    val_losses[k].append(out_loss[k].detach().cpu())
            
            with torch.no_grad():            
                # forward pass to predict srl and/or verbs on the validation set
                verbs_5, pred_SRL_caps, GT_roles, bb_attn =  model.forward_gen(batch)
            
            if self.eval_type == 'Vb_SRL_eval':
                results_vb += self.eval_vb.generate_results(verbs_5, batch)

            res_srl, grounding_list, ious = self.eval_SRL.generate_results(pred_SRL_caps, GT_roles, bb_attn, batch)
            results_SRL += res_srl
            grounding_result += grounding_list

        if self.cfg.train.visualise_bboxes:
            self.dump_groundings(grounding_result, pred_path)

        nums = torch.tensor(nums).float()
        if self.compute_loss:
            val_loss = compute_avg_dict(val_losses, nums)
        
        out_acc = {}
        for results, fname, eval_fn in (list(zip([results_SRL, results_vb], [fname_SRL, fname_vb], [self.eval_SRL.evl_fn, self.eval_vb.evl_fn]))):
            pickle.dump(results, open(fname, "wb"))
            
            if results == []:
                break

            synchronize()
            if is_main_process():
                curr_results = results
                world_size = get_world_size()
                for w in range(1, world_size):
                    tmp_file = Path(pred_path) / f"{dl_name}_{w}.pkl"
                    with open(tmp_file, "rb") as f:
                        tmp_results = pickle.load(f)
                    curr_results += tmp_results
                    tmp_file.unlink
                with open(fname, "wb") as f:
                    pickle.dump(curr_results, f)
                spl = "valid"
                out_acc_cur = eval_fn(fname, split_type=spl)
                out_acc.update(out_acc_cur)
            
        val_acc = {
            k: torch.tensor(v).to(self.device)
            for k, v in out_acc.items()
            if k in self.met_keys
        }
        val_acc["IoU"] = torch.tensor(ious[0]).to(self.device)
        val_acc["IoU_30"] = torch.tensor(ious[1]).to(self.device)
        val_acc["IoU_50"] = torch.tensor(ious[2]).to(self.device)

        synchronize()
        if is_main_process():
            if self.compute_loss:
                return val_loss, val_acc
            else:
                dummy_loss = {k: torch.tensor(0.0).to(self.device) for k in loss_keys}
                return dummy_loss, val_acc
        else:
            return (
                {k: torch.tensor(0.0).to(self.device) for k in loss_keys},
                {k: torch.tensor(0.0).to(self.device) for k in self.met_keys},
            )
        

class EvalB(nn.Module):
    def __init__(self, cfg, comm):
        super().__init__()
        self.cfg = cfg
        self.comm = comm
        self.met_keys = ["Per_Ev_Top_1", "Per_Ev_Top_5", "recall_macro_1_th_9"]
        self.evl_met = EvlFn_Vb(self.cfg, self.comm, self.met_keys)
        self.evl_fn = self.evl_met.simple_acc
        self.compute_loss = True
        return

    def generate_results(self, out_verbs, inp):
        mdl_out_probs = F.softmax(out_verbs, dim=-1)
        mdl_probs_sorted, mdl_ixs_sorted = mdl_out_probs.sort(dim=-1, descending=True)
        # label_lst10 = inp["label_tensor10"]
        ann_lst = inp["vseg_idx"]
        topk_save = 5

        def get_dct(pred_vbs, pred_scores, ann_idx):
            pred_vbs_out = []
            pred_scores_out = []
            assert len(pred_vbs) == 5
            assert len(pred_scores) == 5

            # iterate over Ev1-5
            for pvb, pvs in zip(pred_vbs, pred_scores):
                pvb_used = pvb[:topk_save]
                pvb_str = [self.comm.vb_id_vocab.symbols[pv] for pv in pvb_used]
                pred_vbs_out.append(pvb_str)

                pvb_score = pvs[:topk_save]
                pred_scores_out.append(pvb_score)

            return {
                "pred_vbs_ev": pred_vbs_out,
                "pred_scores_ev": pred_scores_out,
                "ann_idx": ann_idx,
            }

        out_dct_lst = [
            get_dct(pred_vbs, pred_scores, ann_idx)
            for pred_vbs, pred_scores, ann_idx in zip(
                mdl_ixs_sorted.tolist(), mdl_probs_sorted.tolist(), ann_lst.tolist(),
            )
        ]
        return out_dct_lst


class EvalB_Gen():
    def __init__(self, cfg, comm):
        self.cfg = cfg
        self.comm = comm
        self.gr_metric = grounding_metric(comm)  
        self.in_met_keys = ["cider", "bleu", "rouge"]        
        self.evl_met = EvalFnCap(
            self.cfg, self.comm, self.in_met_keys, read_val_file=True
        )
        self.evl_fn = self.evl_met.eval_cap_mets
        self.compute_loss = True

        split_files_cfg = self.cfg.ds.vsitu.split_files_lb
        self.vseg_lst = read_file_with_assertion(split_files_cfg['valid'])        


    def generate_results(self, out_sents, args_all, bb_attn, inp):

        ann_lst = inp["vseg_idx"]
        vb_vocab = self.comm.vb_id_vocab
        wvoc = self.comm.gpt2_hf_tok
        arg_role_voc = self.comm.arg_role_vocab
        
        list_res_batch = []
        list_res_for_grounding = []

        num_args = -1
        for B_i, Batch in enumerate(args_all): 
            #Batch = 5x6
            batch_dct = {} 
            list_grounding_text = []
            batch_dct['vb_output'] = {}
            batch_dct['ann_idx'] = ann_lst[B_i].item()

            for ev_i, ev in enumerate(Batch):
                vb_id = inp['verbs_all_ev'][B_i][ev_i]
                verb = vb_vocab[vb_id.item()] 
                ev_i+=1
                key_ev = "Ev{}".format(ev_i)
                batch_dct['vb_output'][key_ev] = {}
                batch_dct['vb_output'][key_ev]['vb_id'] = verb

                for arg in ev:
                    if arg.item() != arg_role_voc.pad_idx:
                        arg_name = arg_role_voc.idx2arg[arg.item()] 
                        num_args+=1
                        args_res_ = out_sents[num_args].cpu().tolist() #[15]
                        eos_idx = args_res_.index(wvoc.eos_token_id)
                        args_res_ = args_res_[0:eos_idx]
                        args_res = wvoc.decode(args_res_, skip_special_tokens=True) #Str
                        batch_dct['vb_output'][key_ev][arg_name] = args_res
                        list_grounding_text.append(str(key_ev+'_'+arg_name+': '+args_res))

                    elif arg.item() == arg_role_voc.pad_idx:
                        break

            list_res_batch.append(batch_dct)
            list_res_for_grounding.append(list_grounding_text)
        
        grounding_info_list = self.get_grounding_info(inp, bb_attn, list_res_for_grounding)
        self.gr_metric.calc_batch_grounding_metric(grounding_info_list, inp)
        ious = self.gr_metric.get_ious()
        return list_res_batch, grounding_info_list, ious


    def get_grounding_info(self, inp, bb_attn, list_res_for_grounding_batch):

        bb_attn = bb_attn.cpu()

        grounding_list = []
        vid_id = inp['vseg_idx'] #B
        vid_seg_id_per_batch = [self.vseg_lst[idx_vid] for idx_vid in vid_id]

        selected_roles_pad_mask = inp['args_len_all_ev'] #Bx5x1x6
        B,N,N_,R = selected_roles_pad_mask.shape
        roles_mask_per_batch = selected_roles_pad_mask.squeeze().view(B,30).bool()

        #new_addition
        B, NR = roles_mask_per_batch.size()
        coords_per_batch = inp['boxes_11_frames'] #Bx11x15x4
        coords_per_batch = coords_per_batch.view(B, 165, 4)
        img_dim_per_batch = inp['img_size'] #Bx11x2 

        for idx, boxes_idx_per_vid in enumerate(bb_attn):
            grounding_dict = {}
            frame_list = []
            box_coord_list = []
            caption_list = []

            vid_seg_id = vid_seg_id_per_batch[idx]
            grounding_dict['vid_id'] = vid_seg_id
            roles_mask = roles_mask_per_batch[idx]
            coords_per_vid = coords_per_batch[idx]
            img_dim_per_vid = img_dim_per_batch[idx]
            result_vid = list_res_for_grounding_batch[idx]
            boxes_idx_selected = boxes_idx_per_vid[roles_mask]
            for idx_arg, box_idx in enumerate(boxes_idx_selected):
                frame = int(box_idx//15)
                frame_name = 'frame_'+str(frame).zfill(2)+'.jpg'
                frame_dir_pth = self.cfg.ds.vsitu.frames_11_dir 
                frame_path = os.path.join(frame_dir_pth, grounding_dict['vid_id'], frame_name)        
                frame_list.append(frame_path)

                box_coord=coords_per_vid[box_idx] #4 x_left, y_top, x_right, y_bottom           
                box_coord_list.append(box_coord)                
                caption_list.append(result_vid[idx_arg])
            
            grounding_dict['frame_list']=frame_list
            grounding_dict['box_coord_list']=box_coord_list
            grounding_dict['caption_list']=caption_list
            grounding_list.append(grounding_dict)

        return grounding_list

                        
class grounding_metric():
    def __init__(self, comm):
        self.stats_iou_raw = 0
        self.stats_iou_50 = 0
        self.stats_iou_30 = 0
        self.total_vid_stats = 0
        self.arg_role_vocab = comm.arg_role_vocab
        self.pad_idx = self.arg_role_vocab.pad_idx
    
    def reset(self):
        self.stats_iou_raw = 0
        self.stats_iou_50 = 0
        self.stats_iou_30 = 0
        self.total_vid_stats = 0

    def get_ious(self):
        iou_50 = self.stats_iou_50/self.total_vid_stats
        iou_30 = self.stats_iou_30/self.total_vid_stats
        iou_raw = self.stats_iou_raw/self.total_vid_stats
        return(iou_raw, iou_30, iou_50)
    
    def get_iou(self, bb1_list, bb2_list):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """

        bb1 = {'x1':bb1_list[0], 'y1':bb1_list[1], 'x2':bb1_list[2], 'y2':bb1_list[3]}
        bb2 = {'x1':bb2_list[0], 'y1':bb2_list[1], 'x2':bb2_list[2], 'y2':bb2_list[3]}
       
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou


    def calc_iou_over_all_GTs(self, pred_box, GT_boxes_list):
        iou_list = []
        for gt_box in GT_boxes_list:
            iou_list.append(self.get_iou(pred_box, gt_box))
        iou = max(iou_list)
        return iou
    
    def calc_iou_stats(self, eval_data):
        
        iou_avg_per_vid = 0
        iou_30_avg_per_vid = 0
        iou_50_avg_per_vid = 0
        for idx_role, pred_dict_per_role in enumerate(eval_data):
            GT_boxes = []
            pred_box = pred_dict_per_role['box_pred']
            pred_frame_num = int(pred_dict_per_role['frame_pred'].split('/')[-1].split('.')[0].split('_')[-1])
            if pred_frame_num in pred_dict_per_role['frames_GT']:
                for idx_frame, frame in enumerate(pred_dict_per_role['frames_GT']):
                    if frame == pred_frame_num:
                        GT_boxes.append(pred_dict_per_role['boxes_GT'][idx_frame])

                iou_per_role = self.calc_iou_over_all_GTs(pred_box, GT_boxes)
                iou_avg_per_vid+=iou_per_role
                if iou_per_role>0.3:
                    iou_30_avg_per_vid+=1
                if iou_per_role>0.5:
                    iou_50_avg_per_vid+=1
            else:
                iou_per_role = 0
  
        iou_avg_per_vid/=(idx_role+1)

        iou_30_avg_per_vid/=(idx_role+1)
        iou_50_avg_per_vid/=(idx_role+1)

        self.stats_iou_raw += iou_avg_per_vid
        self.stats_iou_30 += iou_30_avg_per_vid
        self.stats_iou_50 += iou_50_avg_per_vid
        self.total_vid_stats += 1


    def calc_batch_grounding_metric(self, grounding_info_list, inp):
        gt_frames_all = inp['gt_frames_all_event'].clone() #Bx5x6x11
        gt_boxes_all = inp['gt_boxes_all_event'].clone() #Bx5x6x11x4
        gt_frames_all_mask = inp['gt_frames_all_mask'].clone().bool() #Bx5x6x11
        roles_all_mask = inp['args_len_all_ev'].clone().bool() #Bx5x1x6

        B,E,R,F,C = gt_boxes_all.size()
        roles_all_mask = roles_all_mask.squeeze().view(B,-1) #Bx30
        gt_frames_all = gt_frames_all.view(B, E*R, F) #Bx30x11
        gt_boxes_all = gt_boxes_all.view(B, E*R, F, C) #Bx30x11x4
        gt_frames_all_mask = gt_frames_all_mask.view(B, E*R, F) #Bx30x11


        for b_idx in range(B):
            role_selected_gt_frames = gt_frames_all[b_idx][roles_all_mask[b_idx]] #selected_roles x 11
            role_selected_gt_boxes = gt_boxes_all[b_idx][roles_all_mask[b_idx]] #selected_roles x 11 x 4
            gt_frames_selected_mask = gt_frames_all_mask[b_idx][roles_all_mask[b_idx]] #selected_roles x 11
            grounding_pred = grounding_info_list[b_idx]

            eval_data_per_vid = []
            for idx_role, frames_per_role in enumerate(role_selected_gt_frames):
                pred_frame = grounding_pred['frame_list'][idx_role]
                pred_box = grounding_pred['box_coord_list'][idx_role]
                mask_frames_per_role = gt_frames_selected_mask[idx_role]
                for frame in frames_per_role:
                    if frame != self.pad_idx:
                        frames_selected = frames_per_role[mask_frames_per_role].tolist()
                        boxes_selected = role_selected_gt_boxes[idx_role][mask_frames_per_role].tolist()
                        dict_eval = {'frame_pred': pred_frame, 'frames_GT': frames_selected, 'box_pred': pred_box,'boxes_GT': boxes_selected} 
                        eval_data_per_vid.append(dict_eval)
                        break
                    else:
                        continue

            if eval_data_per_vid!=[]:
                self.calc_iou_stats(eval_data_per_vid)


