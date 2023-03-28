import json
from pathlib import Path
from yacs.config import CfgNode as CN
from utils._init_stuff import yaml
from typing import Dict, Any
import argparse

class CfgProcessor:
    def __init__(self, cfg_pth):
        assert Path(cfg_pth).exists()
        self.cfg_pth = cfg_pth

    def get_vsitu_default_cfg(self):
        with open(self.cfg_pth) as f:
            c4 = yaml.safe_load(f)
        cfg_dct = c4.copy()
        return CN(cfg_dct)

    def get_key_maps(self):
        key_maps = {}
        return key_maps

    @staticmethod
    def get_val_from_cfg(cfg, key_str):
        key_split = key_str.split(".")
        d = cfg
        for k in key_split[:-1]:
            d = d[k]

        return d[key_split[-1]]

    def create_from_dict(self, dct: Dict[str, Any], prefix: str, cfg: CN):
        """
        Helper function to create yacs config from dictionary
        """
        dct_cfg = CN(dct, new_allowed=True)
        prefix_list = prefix.split(".")
        d = cfg
        for pref in prefix_list[:-1]:
            assert isinstance(d, CN)
            if pref not in d:
                setattr(d, pref, CN())
            d = d[pref]
        if hasattr(d, prefix_list[-1]):
            old_dct_cfg = d[prefix_list[-1]]
            dct_cfg.merge_from_other_cfg(old_dct_cfg)

        setattr(d, prefix_list[-1], dct_cfg)
        return cfg

    @staticmethod
    def update_one_full_key(cfg: CN, dct, full_key, val=None):
        if cfg.key_is_deprecated(full_key):
            return
        if cfg.key_is_renamed(full_key):
            cfg.raise_key_rename_error(full_key)

        if val is None:
            assert full_key in dct
            v = dct[full_key]
        else:
            v = val
        key_list = full_key.split(".")
        d = cfg
        for subkey in key_list[:-1]:
            # Most important statement
            assert subkey in d, f"key {full_key} doesnot exist"
            d = d[subkey]

        subkey = key_list[-1]
        # Most important statement
        assert subkey in d, f"key {full_key} doesnot exist"

        value = cfg._decode_cfg_value(v)

        assert isinstance(value, type(d[subkey]))
        d[subkey] = value

        return

    def update_from_dict(
        self, cfg: CN, dct: Dict[str, Any], key_maps: Dict[str, str] = None
    ) -> CN:
        """
        Given original CfgNode (cfg) and input dictionary allows changing
        the cfg with the updated dictionary values
        Optional key_maps argument which defines a mapping between
        same keys of the cfg node. Only used for convenience
        Adapted from:
        https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L219
        """
        # Original cfg
        # root = cfg
        if key_maps is None:
            key_maps = []
        # Change the input dictionary using keymaps
        # Now it is aligned with the cfg
        full_key_list = list(dct.keys())
        for full_key in full_key_list:
            if full_key in key_maps:
                # cfg[full_key] = dct[full_key]
                self.update_one_full_key(cfg, dct, full_key)
                new_key = key_maps[full_key]
                # dct[new_key] = dct.pop(full_key)
                self.update_one_full_key(cfg, dct, new_key, val=dct[full_key])

        # Convert the cfg using dictionary input
        # for full_key, v in dct.items():
        for full_key in dct.keys():
            self.update_one_full_key(cfg, dct, full_key)
        return cfg

    @staticmethod
    def pre_proc_config(cfg: CN):
        """
        Add any pre processing based on cfg
        """
        return cfg

    @staticmethod
    def post_proc_config(cfg: CN):
        """
        Add any post processing based on cfg
        """
        return cfg

    @staticmethod
    def cfg_to_flat_dct(cfg: CN):
        def to_flat_dct(dct, prefix_key: str):
            def get_new_key(prefix_key, curr_key):
                if prefix_key == "":
                    return curr_key
                return prefix_key + "." + curr_key

            out_dct = {}
            for k, v in dct.items():
                if isinstance(v, dict):
                    out_dct1 = to_flat_dct(v, prefix_key=get_new_key(prefix_key, k))
                else:
                    out_dct1 = {get_new_key(prefix_key, k): v}
                out_dct.update(out_dct1)
            return out_dct

        cfg_dct = json.loads(json.dumps(cfg))
        return to_flat_dct(cfg_dct, prefix_key="")

    @staticmethod
    def to_str(cfg: CN):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(cfg.items()):
            # seperator = "\n" if isinstance(v, CN) else " "
            if isinstance(v, CN):
                seperator = "\n"
                str_v = CfgProcessor.to_str(v)
            else:
                seperator = " "
                str_v = str(v)
                if str_v == "" or str_v == "":
                    str_v = "''"
            attr_str = "{}:{}{}".format(str(k), seperator, str_v)
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r
