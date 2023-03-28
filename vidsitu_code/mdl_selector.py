from vidsitu_code.ground_vsitu import Grounded_VidSitu, LossLambda
from vidsitu_code.evl_vsitu import Eval_grvidsitu


def get_mdl_loss_eval(cfg):    
    if cfg.task_type == "grounded_srl_GT_vbrole" :
        cfg.defrost()
        cfg.loss_keys = ['loss_SRL']
        cfg.eval_type = 'SRL_eval'
        cfg.freeze() 
        return {"mdl": Grounded_VidSitu, "loss": LossLambda, "evl": Eval_grvidsitu}

    elif cfg.task_type == "grounded_vb_srl_GT_role" :
        cfg.defrost()
        cfg.loss_keys = ['loss_SRL', 'loss_vb']
        cfg.eval_type = 'Vb_SRL_eval'
        cfg.freeze()
        return {"mdl": Grounded_VidSitu, "loss": LossLambda, "evl": Eval_grvidsitu}

    elif cfg.task_type == "grounded_end-to-end" :
        cfg.defrost()
        cfg.loss_keys = ['loss_SRL','loss_vb','loss_role']
        cfg.eval_type = 'Vb_SRL_eval'
        cfg.freeze()        
        return {"mdl": Grounded_VidSitu, "loss": LossLambda, "evl": Eval_grvidsitu}    
