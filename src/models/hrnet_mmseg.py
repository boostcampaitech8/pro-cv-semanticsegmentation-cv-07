from mmengine.config import Config
from mmseg.models import build_segmentor
from mmengine.runner import load_checkpoint

def get_mmseg_model(cfg_path, num_classes):
    cfg = Config.fromfile(cfg_path)
    
    if isinstance(cfg.model.decode_head, list):
        for head in cfg.model.decode_head:
            head.num_classes = num_classes
    else:
        cfg.model.decode_head.num_classes = num_classes

    model = build_segmentor(cfg.model)
    
    model.init_weights()
    
    return model
