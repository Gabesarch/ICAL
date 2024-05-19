import os
from os import path
from argparse import ArgumentParser

import torch
# from torch.utils.data import DataLoader
import numpy as np

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args
from deva.ext.automatic_sam import get_sam_model
from deva.ext.automatic_processor import process_frame_automatic as process_frame

from tqdm import tqdm
import json
import ipdb
st = ipdb.set_trace
from dataclasses import dataclass
import ipdb
st = ipdb.set_trace
# if __name__ == '__main__':
torch.autograd.set_grad_enabled(False)

# @dataclass
class ARGS_SOLQ():
    def __init__(self):
        # default params
        self.size: int = 480
        self.chunk_size: int = 1
        self.mem_every: int = 5
        self.top_k: int = 30
        self.num_prototypes: int = 128
        self.max_long_term_elements: int = 10000
        self.min_mid_term_frames: int = 5
        self.max_mid_term_frames: int = 10
        self.disable_long_term: bool = False
        self.pix_feat_dim: int = 512
        self.value_dim: int = 512
        self.key_dim: int = 64
        self.amp: bool = False
        self.save_all: bool = False
        self.output: str = 'Tracking-Anything-with-DEVA/example'
        self.model: str = 'Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth'
        self.GROUNDING_DINO_CONFIG_PATH: str = 'Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py'
        self.GROUNDING_DINO_CHECKPOINT_PATH: str = 'Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth'
        self.DINO_THRESHOLD: float = 0.35
        self.DINO_NMS_THRESHOLD: float = 0.8
        self.SAM_ENCODER_VERSION: str = 'vit_h'
        self.SAM_CHECKPOINT_PATH: str = 'Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth'
        self.MOBILE_SAM_CHECKPOINT_PATH: str = 'Tracking-Anything-with-DEVA/saves/mobile_sam.pt'
        self.SAM_NUM_POINTS_PER_SIDE: int = 64
        self.SAM_NUM_POINTS_PER_BATCH: int = 64
        self.SAM_PRED_IOU_THRESHOLD: float = 0.88
        self.SAM_OVERLAP_THRESHOLD: float = 0.8
        self.img_path: str = 'Tracking-Anything-with-DEVA/example/vipseg'
        self.detection_every: int = 5
        self.num_voting_frames: int = 3
        self.temporal_setting: str = 'semionline'  # 'online'
        self.max_missed_detection_count: int = 5
        self.max_num_objects: int = 200
        self.sam_variant: str = 'original'  # 'mobile'
        self.suppress_small_objects: bool = True

class DEVA:
    def __init__(self):

        # # for id2rgb
        # np.random.seed(42)
        # """
        # Arguments loading
        # """
        # parser = ArgumentParser()
        

        args = ARGS_SOLQ()

        deva_model, cfg, args = get_model_and_config(args)
        sam_model = get_sam_model(cfg, 'cuda')

        """
        Temporal setting
        """
        # cfg = {}
        cfg['temporal_setting'] = args.temporal_setting.lower()
        assert cfg['temporal_setting'] in ['semionline', 'online']

        # get data
        # video_reader = SimpleVideoReader(cfg['img_path'])
        # loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
        

        # Start eval
        # vid_length = len(loader)

        vid_length = 16
        # no need to count usage for LT if the video is not that long anyway
        cfg['enable_long_term_count_usage'] = (
            cfg['enable_long_term']
            and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
                    cfg['num_prototypes']) >= cfg['max_long_term_elements'])

        print('Configuration:', cfg)

        deva = DEVAInferenceCore(deva_model, config=cfg)
        deva.next_voting_frame = args.num_voting_frames - 1
        deva.enabled_long_id()

        self.cfg = cfg
        self.deva = deva
        self.deva_model = deva_model
        self.sam_model = sam_model
        self.args = args

    def reset(self):
        deva = DEVAInferenceCore(self.deva_model, config=self.cfg)
        deva.next_voting_frame = self.args.num_voting_frames - 1
        deva.enabled_long_id()
        self.deva = deva

    def run_deva(self, images):
        self.reset()
        out_path = self.cfg['output']
        result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=self.deva.object_manager)
        with torch.cuda.amp.autocast(enabled=self.args.amp):
            for ti, frame in enumerate(images): #tqdm(loader)):
                process_frame(self.deva, self.sam_model, f"{ti}.png", result_saver, ti, image_np=np.asarray(frame))
            flush_buffer(self.deva, result_saver)
        result_saver.end(anno_mode=['Mark'])

        # # save this as a video-level json
        # with open(path.join(out_path, 'pred.json'), 'w') as f:
        #     json.dump(result_saver.video_json, f, indent=4)  # prettier json
        images = result_saver.images

        # for idx, image in enumerate(images):
        #     image.save(f'images/test{idx}.png')
        # st()
        return images