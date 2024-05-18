import numpy as np
import os
from arguments import args
import random
import math
from tensorboardX import SummaryWriter
import torch
import utils.aithor
import utils.geom
import PIL
from PIL import Image
import ipdb
st = ipdb.set_trace
import matplotlib.pyplot as plt
from task_base.aithor_base import Base
from itertools import cycle
from tqdm import tqdm
# from utils.aithor import get_masks_from_seg
# from utils.ddetr_utils import check_for_detections_local_policy
from task_base.animation_util import Animation
from torchvision import transforms
import torch.nn.functional as F
# from nets.short_term_policy import DDETR
# from backend.dataloaders.RICK_loader_mem import RICKDataset, my_collate
import h5py

import sys
# sys.path.append('alfred')
# sys.path.append('alfred/gen')
# from gen.utils.video_util import VideoSaver
# from gen.utils.py_util import walklevel
# from env.thor_env import ThorEnv
# from gen.augment_traj import get_image_index

sys.path.append(os.path.join(args.alfred_root))
sys.path.append(os.path.join(args.alfred_root, 'gen'))

import json
import glob
import os
import constants
import cv2
import shutil
import numpy as np
import argparse
import threading
import string
import time
import copy
import random
# from alfred.gen.utils.video_util import VideoSaver
from alfred.gen.utils.py_util import walklevel
from alfred.env.thor_env import ThorEnv

from alfred.gen.constants import VAL_RECEPTACLE_OBJECTS

from backend import saverloader

import ai2thor.controller

def noop(self):
    pass
    # self.step(dict(action='Pass'))

ai2thor.controller.Controller.lock_release = noop
ai2thor.controller.Controller.unlock_release = noop
ai2thor.controller.Controller.prune_releases = noop

import pickle    

# TRAJ_DATA_JSON_FILENAME = "traj_data.json"
JSON_FILE = args.json_file #"json_2.1.0"
# AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"
TRAJ_DATA_JSON_FILENAME = "traj_data.json"

ORIGINAL_IMAGES_FOLDER = "raw_images"
HISTORY_IMAGES_FOLDER = "raw_history_images"
HIGH_RES_IMAGES_FOLDER = "high_res_images"
DEPTH_IMAGES_FOLDER = "depth_images"
INSTANCE_MASKS_FOLDER = "instance_masks"
TARGETS_FOLDER = "targets"

IMAGE_WIDTH = args.H
IMAGE_HEIGHT = args.W

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

# video_saver = VideoSaver()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def get_image_index(save_path):
    return len(glob.glob(save_path + '/*.png'))

def get_image_index_from_two(save_path, save_path2):
    return len(glob.glob(save_path + '/*.png')+glob.glob(save_path2 + '/*.png'))

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class Ai2Thor_GEN(Base):

    def __init__(self, lock=None):   

        super(Ai2Thor_GEN, self).__init__()

        self.max_objs = args.max_objs
        # self.only_intermediate_objs_memory = args.only_intermediate_objs_memory

        self.traj_list = []
        # lock = threading.Lock()

        receptacles = list(VAL_RECEPTACLE_OBJECTS.keys())
        objects = [list(l) for l in list(VAL_RECEPTACLE_OBJECTS.values())]
        objects = list(set(sum(objects, [])))
        self.most_common_receptacles = {}
        for obj in objects:
            self.most_common_receptacles[obj] = {}
            for rec in receptacles:
                if obj in VAL_RECEPTACLE_OBJECTS[rec]:
                    self.most_common_receptacles[obj][rec] = 0

        if lock is not None:
            self.lock = lock
        else:
            self.lock = None

        print(f"Max trajectories: {args.max_trajectories}")

        # make a list of all the traj_data json files
        print("Getting files...")

        if args.load_traj_list is not None:
            with open(args.load_traj_list, "rb") as fp:
                self.traj_list = pickle.load(fp)
            return

        time0 = time.time()
        for dir_name, subdir_list, file_list in tqdm(walklevel(args.alfred_data_path, level=3)):
            if "trial_" in dir_name:
                json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
                if not args.agent_evaluation and not os.path.isfile(json_file): # or 'tests' in dir_name or "val" in dir_name:
                    # print('skip 1')
                    continue
                
                # skip splits not specified
                skip = True
                for split_ in args.splits_to_generate:
                    # print(dir_name)
                    if split_ in dir_name:
                        skip = False
                        break
                if skip:
                    # print('skip 2')
                    continue
                
                # print("Processing:", json_file)
                self.traj_list.append(json_file)
            
            # generate a subset of trajs if we want
            if args.max_trajectories is not None:
                # print(len(self.traj_list), args.max_trajectories)
                if len(self.traj_list)>args.max_trajectories:
                    break

        # random shuffle
        if args.shuffle:
            random.shuffle(self.traj_list)

        if args.reverse_order:
            self.traj_list = self.traj_list[::-1]

        time1 = time.time()

        print("time:", time1-time0)

        # # start threads
        # threads = []
        # for n in range(args.num_threads):
        #     thread = threading.Thread(target=run)
        #     threads.append(thread)
        #     thread.start()
        #     time.sleep(1)

    def generate_trajs(self):

        # start THOR env
        print("Server port:", self.server_port)
        print(f"Image size: ({self.W} {self.H})")
        self.env = ThorEnv(player_screen_width=self.W,
                            player_screen_height=self.H,
                            x_display=str(self.server_port),
                            )

        skipped_files = []

        print("Trajectory length:", len(self.traj_list))

        count = -1

        successful_subgoals = []
        successful_subgoals_dict = {
            "PutObject":[], 
            "OpenObject":[], 
            "CloseObject":[], 
            "PickupObject":[], 
            "PutObject":[], 
            "ToggleObjectOn":[], 
            "ToggleObjectOff":[], 
            "SliceObject":[]
            }

        if args.subsample_eval is not None:
            self.traj_list = self.traj_list[::args.subsample_eval]

        # self.traj_list.reverse()
        # while len(self.traj_list) > 0:
        for json_file in tqdm(self.traj_list):
            count += 1
            # if self.lock is not None:
            #     self.lock.acquire()
            # json_file = self.traj_list.pop()
            # if self.lock is not None:
            #     self.lock.release()

            if args.start_index is not None:
                if count<args.start_index:
                    continue

            # if args.skip_if_exists:
            #     save_folder = json_file.split(JSON_FILE+'/')[1]
            #     if os.path.exists(os.path.join(args.alfred_gen_root, save_folder, ORIGINAL_IMAGES_FOLDER)):
            #         continue
            json_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")
            # check if already exists
            save_folder = json_dir.split(JSON_FILE+'/')[1]

            print ("Augmenting: " + json_file)

            if not args.override_existing and os.path.exists(os.path.join(args.alfred_gen_root, save_folder)):
                print(f'{count}: {save_folder} exists.. skipping')
                continue
            if args.in_try_except:
                try:
                    self.augment_traj(self.env, json_file)
                except KeyboardInterrupt:
                    sys.exit(1)
                except Exception as e:
                    print("EXCEPTION:", Exception)
                    print(f"{json_file} failed... skipping in generation...")
            else:
                self.augment_traj(self.env, json_file)

            print(self.most_common_receptacles)

            os.makedirs(args.alfred_gen_root, exist_ok=True)
            map_path = os.path.join(args.alfred_gen_root, 'most_common_receptacle.p')
            print(f"Saving {map_path}...")
            with open(map_path, "wb") as file_:
                pickle.dump(self.most_common_receptacles, file_, -1)

            

        self.env.stop()
        print("Finished.")

        # skipped files
        if len(skipped_files) > 0:
            print("Skipped Files:")
            print(skipped_files)


    def augment_traj(self, env, json_file):
        '''
        max_openable_per_cat: how many max of each category to get observations for
        '''
        
        # load json data
        with open(json_file) as f:
            traj_data = json.load(f)

        # make directories
        root_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")
        json_dir = root_dir

        orig_images_dir = os.path.join(root_dir, ORIGINAL_IMAGES_FOLDER)
        high_res_images_dir = os.path.join(root_dir, HIGH_RES_IMAGES_FOLDER)
        depth_images_dir = os.path.join(root_dir, DEPTH_IMAGES_FOLDER)
        instance_masks_dir = os.path.join(root_dir, INSTANCE_MASKS_FOLDER)
        augmented_json_file = os.path.join(root_dir, AUGMENTED_TRAJ_DATA_JSON_FILENAME)

        # fresh images list
        traj_data['images'] = list()

        # clear_and_create_dir(high_res_images_dir)
        # clear_and_create_dir(depth_images_dir)
        # clear_and_create_dir(instance_masks_dir)

        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        object_toggles = traj_data['scene']['object_toggles']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']

        # reset
        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        env.step(dict(traj_data['scene']['init_action']))
        # print("Task: %s" % (traj_data['template']['task_desc']))

        # setup task
        env.set_task(traj_data, args, reward_type='dense')
        rewards = []

        for object_ in env.last_event.metadata["objects"]:
            obj_type = object_["objectType"]
            if obj_type in self.most_common_receptacles.keys():
                if object_['parentReceptacles'] is not None:
                    for p_rec in object_['parentReceptacles']:
                        rec = p_rec.split('|')[0]
                        if rec=="Sink":
                            rec='SinkBasin'
                        if rec not in self.most_common_receptacles[obj_type].keys():
                            self.most_common_receptacles[obj_type][rec] = 0
                        self.most_common_receptacles[obj_type][rec] += 1

        
