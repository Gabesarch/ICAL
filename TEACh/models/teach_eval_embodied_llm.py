import sys
import json

import ipdb
st = ipdb.set_trace
from arguments import args

from time import sleep
from typing import List
import matplotlib.pyplot as plt
from ai2thor.controller import Controller

from task_base.object_tracker import ObjectTrack
from task_base.navigation import Navigation
from task_base.animation_util import Animation
from task_base.teach_base import TeachTask
from backend import saverloader
import pickle

import numpy as np
import os

import cv2

import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from teach.dataset.dataset import Dataset
from teach.dataset.definitions import Definitions
from teach.logger import create_logger
from teach.simulators import simulator_factory
from teach.utils import get_state_changes, reduce_float_precision
import torch
import utils
import utils.geom
import logging
from teach.replay.episode_replay import EpisodeReplay
if args.mode in ["teach_eval_tfd", "teach_eval_custom", "teach_eval_continual"]:
    from teach.inference.tfd_inference_runner import TfdInferenceRunner as InferenceRunner
elif args.mode=="teach_eval_edh":
    from teach.inference.edh_inference_runner import EdhInferenceRunner as InferenceRunner
else:
    from teach.inference.tfd_inference_runner import TfdInferenceRunner as InferenceRunner
    # assert(False) # what mode is this? 
from teach.inference.edh_inference_runner import InferenceRunnerConfig
from teach.utils import (
    create_task_thor_from_state_diff,
    load_images,
    save_dict_as_json,
    with_retry,
    load_json
)
from teach.eval.compute_metrics import create_new_traj_metrics, evaluate_traj
from prompt.run_gpt import LLMPlanner
import copy
import traceback

from task_base.aithor_base import get_rearrangement_categories

logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
                filename='./subgoalcontroller.log',
                filemode='w'
            )

from IPython.core.debugger import set_trace
from PIL import Image
import wandb

from .agent.executor import ExecuteController
from .agent.planner import PlannerController

from .agent.api_primitives_executable import InteractionObject, CustomError
from .agent.api_primitives_corrective import AgentCorrective
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)

class SubGoalController(ExecuteController, PlannerController):
    def __init__(
        self, 
        data_dir: str, 
        output_dir: str, 
        images_dir: str, 
        edh_instance: str = None, 
        max_init_tries: int =5, 
        replay_timeout: int = 500, 
        num_processes: int = 1, 
        iteration=0,
        er=None,
        depth_network=None,
        segmentation_network=None,
        attribute_detector=None,
        clip=None,
        ) -> None:

        super(SubGoalController, self).__init__()

        self.er = er
        split = args.split

        ###########%%%%%% PARAMS %%%%%%%#########
        keep_head_down = True # keep head oriented dwon when navigating (want this True if estimating depth)
        keep_head_straight = False # keep head oriented straight when navigating (want this False if estimating depth)

        self.log_every = args.log_every # log every X iters if generating video
        # self.eval_rows_to_add = eval_rows_to_add

        self.teleport_to_objs = args.teleport_to_objs # teleport to objects instead of navigating
        self.render = args.create_movie and (iteration % self.log_every == 0) # render video? NOTE: video is rendered to self.root folder
        use_gt_objecttrack = args.use_gt_seg # if navigating, use GT object masks for getting object detections + centroids?
        use_gt_depth = args.use_gt_depth # if navigating, use GT depth maps? 
        self.use_GT_seg_for_interaction = args.use_gt_seg # use GT seg for interaction? 
        self.use_GT_constraint_checks = args.use_GT_constraint_checks
        self.use_GT_error_feedback = args.use_GT_error_feedback
        self.use_gt_subgoals = args.use_gt_subgoals
        self.use_llm_search = args.use_llm_search
        self.use_progress_check = args.use_progress_check
        self.add_back_objs_progresscheck = args.add_back_objs_progresscheck
        self.use_constraint_check = args.use_constraint_check
        self.use_mask_rcnn_pred = args.use_mask_rcnn_pred
        self.environment_index = 0
        self.log = True
        # self.output_folder = f'output/skill_logging_{args.set_name}'
        if args.mode=="teach_eval_tfd":
            self.output_folder = f'output/eval_log/{args.mode}_{args.set_name}'
        else:
            self.output_folder = f'output/{args.mode}_{args.set_name}'
        os.makedirs(os.path.join(self.output_folder, 'logging'), exist_ok=True)
        # self.demo_folder = args.demo_folder
        self.folder_tag = f'{os.path.split(edh_instance)[-1].split(".tfd")[0]}'
        os.makedirs(os.path.join(self.output_folder, 'logging', self.folder_tag), exist_ok=True) 


        do_masks = args.do_masks # use masks from detector. If False, use boxes (Note: use_gt_objecttrack must be False)
        use_solq = args.use_solq # use SOLQ model? need this for masks
        
        self.new_parser = args.new_parser

        
        self.episode_in_try_except = args.episode_in_try_except # Continue to next episode if assertion error occurs? 

        self.dist_thresh = args.dist_thresh # distance threshold for point goal navigation 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mode = args.mode

        if self.teleport_to_objs:
            self.use_GT_seg_for_interaction = True
            self.add_map = False
        else:
            self.add_map = True

        self.interacted_ids = [] # for teleporting to keep track
        self.navigation_action_fails = 0
        self.obj_center_camX0 = None
        self.err_message = ''
        self.help_message = ''

        self.failed_subgoals = []
        self.successful_subgoals = []
        self.error_correct_fix = []
        self.attempted_subgoals = []
        self.llm_log = {"Dialogue":"", "LLM output":"", "subgoals":"", "full_prompt":""}
        
        self.traj_steps_taken: int = 0
        self.iteration = iteration
        self.replan_num = 0
        self.num_subgoal_fail = 0
        self.max_subgoal_fail = 10
        self.errors = []
        self.could_not_find = []
        self.completed_subgoals = []
        self.object_tracker_ids_removed = []
        self.run_error_correction_llm = args.run_error_correction_llm
        self.run_error_correction_basic = args.run_error_correction_basic
        self.visibility_distance = args.visibilityDistance
        self.edh_instance_file = edh_instance
        self.step_success = True
        self.replan_number = 1

        self.action_to_mappedaction = {
            'MoveAhead':"Forward", 
            "RotateRight":"Turn Right", 
            "RotateLeft":"Turn Left", 
            "LookDown":"Look Down",
            "LookUp":"Look Up",
            "Done":"Stop",
            'PutObject':"Place",
            'PickupObject':"Pickup",
            }

        self.include_classes = [
            'ShowerDoor', 'Cabinet', 'CounterTop', 'Sink', 'Towel', 'HandTowel', 'TowelHolder', 'SoapBar', 
            'ToiletPaper', 'ToiletPaperHanger', 'HandTowelHolder', 'SoapBottle', 'GarbageCan', 'Candle', 'ScrubBrush', 
            'Plunger', 'SinkBasin', 'Cloth', 'SprayBottle', 'Toilet', 'Faucet', 'ShowerHead', 'Box', 'Bed', 'Book', 
            'DeskLamp', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'Painting', 'CreditCard', 
            'AlarmClock', 'CD', 'Laptop', 'Drawer', 'SideTable', 'Chair', 'Blinds', 'Desk', 'Curtains', 'Dresser', 
            'Watch', 'Television', 'WateringCan', 'Newspaper', 'FloorLamp', 'RemoteControl', 'HousePlant', 'Statue', 
            'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'BaseballBat', 'TennisRacket', 'VacuumCleaner', 'Mug', 'ShelvingUnit', 
            'Shelf', 'StoveBurner', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Microwave', 'CoffeeMachine', 'Fork', 'Fridge', 
            'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 
            'ButterKnife', 'StoveKnob', 'Toaster', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'DiningTable', 'Bowl', 
            'LaundryHamper', 'Vase', 'Stool', 'CoffeeTable', 'Poster', 'Bathtub', 'TissueBox', 'Footstool', 'BathtubBasin', 
            'ShowerCurtain', 'TVStand', 'Boots', 'RoomDecor', 'PaperTowelRoll', 'Ladle', 'Kettle', 'Safe', 'GarbageBag', 'TeddyBear', 
            'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window', 'LightSwitch']
        self.special_classes = ['AppleSliced', 'BreadSliced', 'EggCracked', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']
        self.include_classes += self.special_classes
        if self.use_mask_rcnn_pred:
            self.include_classes += ['Cart', 'PaintingHanger', 'Glassbottle', 'LaundryHamperLid', 'PaperTowel', 'ToiletPaperRoll']
        self.include_classes.append('no_object') # ddetr has no object class

        # user defined action list
        self.NONREMOVABLE_CLASSES = ["Toilet", "Desk", "StoveKnob", "Faucet", "Fridge", "SinkBasin", "Sink", "Bed", "Microwave", "CoffeeTable", "HousePlant", "DiningTable", "Sofa", 'ArmChair', 'Toaster', 'CoffeeMachine', 'Lettuce', 'Tomato', 'Bread', 'Potato', 'Plate']
        self.FILLABLE_CLASSES = ["Bottle", "Bowl", "Cup", "HousePlant", "Kettle", "Mug", "Pot", "WateringCan", "WineBottle"]
        self.SLICEABLE = [
            'Apple', 
            'Bread', 
            'Lettuce', 
            'Potato', 
            'Tomato', 
            # 'AppleSliced', 
            # 'BreadSliced', 
            # 'LettuceSliced', 
            # 'PotatoSliced', 
            # 'TomatoSliced'
            ]
        self.TOASTABLE = ['Bread', 'BreadSliced']
        self.TOGGLEABLE = ['DeskLamp', 'FloorLamp', 'StoveKnob', 'Microwave', 'Toaster', 'Faucet', 'CoffeeMachine']
        self.DIRTYABLE = ['Apple',
                #   'AppleSliced',
                  'Bowl',
                #   'ButterKnife',
                  'Cloth',
                  'Cup',
                  'DishSponge',
                #   'Egg',
                  'Fork',
                  'Kettle',
                #   'Knife',
                  'Ladle',
                #   'Lettuce',
                #   'LettuceSliced',
                  'Mug',
                  'Pan',
                  'Plate',
                  'Pot',
                #   'Potato',
                #   'PotatoSliced',
                  'SoapBar',
                  'Spatula',
                  'Spoon',
                #   'Tomato',
                #   'TomatoSliced'
                  ]
        self.COOKABLE = [
                 'Apple',
                 'AppleSliced',
                 'Bread',
                 'BreadSliced',
                 'Potato',
                 'PotatoSliced',
                 'Tomato',
                 'TomatoSliced']

        # self.EMPTYABLE = [
        #     'SinkBasin',
        #     'Plate',
        #     'Pan',
        #     'BathtubBasin',

        self.EMPTYABLE = [
            # 'ArmChair', 
            # 'BathtubBasin', 
            'Bowl', 
            'Box', 
            # 'Cabinet', 
            # 'Chair', 
            'CoffeeMachine', 
            'Cup', 
            # 'Drawer', 
            # 'Footstool', 
            # 'GarbageCan', 
            # 'HandTowelHolder', 
            # 'LaundryHamper', 
            'Microwave', 
            'Mug', 
            # 'Ottoman', 
            'Pan', 
            'Plate', 
            'Pot', 
            # 'Safe', 
            # 'Shelf', 
            # 'ShelvingUnit', 
            # 'SideTable', 
            'SinkBasin', 
            # 'Sofa', 
            # 'Stool', 
            'StoveBurner', 
            # 'TVStand', 
            # 'Toaster', 
            # 'Toilet', 
            # 'ToiletPaperHanger', 
            # 'TowelHolder'
            ]
        
        # self.RETRY_ACTIONS = ["Pickup", "Place", "Slice", "Pour"]
        self.RETRY_ACTIONS_IMAGE = ["Place", "Pour"]
        self.RETRY_DICT_IMAGE = {a:self.include_classes for a in self.RETRY_ACTIONS_IMAGE}
        self.RETRY_DICT_IMAGE["Place"] = ["CounterTop", "Bed", "DiningTable", "CoffeeTable", "SinkBasin", "Sink", "Sofa", 'ArmChair', 'Plate', 'Bowl']
        # self.RETRY_DICT_IMAGE["Pour"] = ["HousePlant"]
        self.RETRY_ACTIONS_LOCATION = ["Pickup", "Place", "Slice", "Pour", "ToggleOn", "ToggleOff", "Open", "Close"]
        self.RETRY_DICT_LOCATION = {a:self.include_classes for a in self.RETRY_ACTIONS_LOCATION}
        self.OPENABLE_CLASS_LIST = set(['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box'])

        _, _, self.PICKUPABLE_OBJECTS, self.OPENABLE_OBJECTS, self.RECEPTACLE_OBJECTS = get_rearrangement_categories()
        self.PICKUPABLE_OBJECTS += self.special_classes

        self.general_receptacles_classes = [
                'CounterTop', 'DiningTable', 'CoffeeTable', 'SideTable',
                'Desk', 'Bed', 'Sofa', 'ArmChair',
                'Chair', 'Dresser', 'Ottoman', 'DogBed', 
                'Footstool', 'Safe', 'TVStand'
                ]
        self.clean_classes = ["Bowl", "Cup", "Mug", "Plate"]

        if not self.use_gt_subgoals:
            self.llm = LLMPlanner(
                args.gpt_embedding_dir, 
                fillable_classes=self.FILLABLE_CLASSES, 
                openable_classes=self.OPENABLE_CLASS_LIST,
                include_classes=self.include_classes,
                clean_classes=self.clean_classes,
                example_mode=args.mode,
                )

        if not self.use_gt_subgoals and clip is None:
            from nets.clip import CLIP
            self.clip = CLIP()  
        else:
            self.clip = clip

        if args.get_skills_demos:
            return

        self.name_to_id = {}
        self.id_to_name = {}
        self.instance_counter = {}
        idx = 0
        for name in self.include_classes:
            self.name_to_id[name] = idx
            self.id_to_name[idx] = name
            self.instance_counter[name] = 0
            idx += 1

        if self.use_GT_seg_for_interaction:
            # self.name_to_mapped_name = {'DeskLamp':'FloorLamp', 'EggCracked':'Egg'} # 'Sink':'SinkBasin', 'Bathtub':'BathtubBasin', 
            self.name_to_mapped_name = {'DeskLamp':'FloorLamp', 'Sink':'SinkBasin', 'Bathtub':'BathtubBasin', 'EggCracked':'Egg', 'ArmChair':'Chair', 'ButterKnife': 'Knife'}
            self.id_to_mapped_id = {self.name_to_id[k]:self.name_to_id[v] for k, v in self.name_to_mapped_name.items()}
        else:
            self.name_to_mapped_name = {'DeskLamp':'FloorLamp', 'Sink':'SinkBasin', 'Bathtub':'BathtubBasin', 'EggCracked':'Egg', 'ArmChair':'Chair', 'ButterKnife': 'Knife'} #, 'AppleSliced':'Apple', 'BreadSliced':'Bread', 'EggCracked':'Egg', 'LettuceSliced':'Lettuce', 'PotatoSliced':'Potato', 'TomatoSliced':'Tomato'}
            if self.use_mask_rcnn_pred:
                self.name_to_mapped_name.update({'Cart':'Desk', 'PaintingHanger':'Painting', 'Glassbottle':'Bottle', 'LaundryHamperLid':'LaundryHamper','PaperTowel':'PaperTowelRoll', 'ToiletPaperRoll':'ToiletPaper'})
            self.id_to_mapped_id = {self.name_to_id[k]:self.name_to_id[v] for k, v in self.name_to_mapped_name.items()}

        self.name_to_mapped_name_subgoals = {'DeskLamp':'FloorLamp', 'Sink':'SinkBasin', 'Bathtub':'BathtubBasin', 'EggCracked':'Egg', 'Cart':'Desk', 'PaintingHanger':'Painting', 'Glassbottle':'Bottle', 'LaundryHamperLid':'LaundryHamper','PaperTowel':'PaperTowelRoll', 'ToiletPaperRoll':'ToiletPaper'}
        self.id_to_mapped_id_subgoals = {self.name_to_id[k]:self.name_to_id[v] for k, v in self.name_to_mapped_name.items()}

        self.W = args.W
        self.H = args.H
        self.web_window_size = args.W
        self.fov = args.fov
        print(f"fov: {self.fov}")
        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.

        self.navigate_obj_info = {}
        self.navigate_obj_info["object_class"] = None
        self.navigate_obj_info["object_center"] = None
        self.navigate_obj_info["obj_ID"] = None
        
        self.runner_config = InferenceRunnerConfig(
            data_dir=data_dir,
            split=split,
            output_dir=output_dir,
            images_dir=images_dir,
            model_class=None,
            model_args=None,
            num_processes=num_processes,
            max_init_tries=max_init_tries,
            replay_timeout=replay_timeout,
            max_api_fails=args.max_api_fails,
            max_traj_steps=args.max_traj_steps,
        )
        print("INIT TfD...")
        self.init_success = self.load_edh_instance(edh_instance)

        self.teach_task = TeachTask(
            self.er,
            action_to_mappedaction=self.action_to_mappedaction,
            approx_last_action_success=not args.use_gt_success_checker,
            max_fails=self.runner_config.max_api_fails,
            max_steps=self.runner_config.max_traj_steps,
            remove_unusable_slice=args.remove_unusable_slice,
            use_GT_error_feedback=self.use_GT_error_feedback,
        )
        self.teach_task.metrics = self.metrics

        if not self.init_success:
            print("task initialization failed.. moving to next episode..")
            return 
        self.controller = self.er.simulator.controller
        self.tag = f"{self.instance_id}_{self.game_id}"
        print("DONE.")
        
        # Find out if the agent is holding something to start 
        object_cat_pickup = None
        action_history = self.edh_instance['driver_action_history']
        in_hand = False
        for action in action_history:
            if action['action_name']=='Pickup' and action['oid'] is not None:
                object_id_pickup = action['oid']
                object_cat_pickup = object_id_pickup.split('|')[0]
                in_hand = True
            if action['action_name']=='Place' and action['oid'] is not None:
                object_id_pickup = None
                object_cat_pickup = None
                in_hand = False

        keep_head_straight = False
        search_pitch_explore = False
        block_map_positions_if_fail=True
        if args.use_estimated_depth or args.increased_explore:
            look_down_init = True
            keep_head_down = False
        else:
            look_down_init = True
            keep_head_down = False
        self.navigation = Navigation(
            controller=self.controller, 
            keep_head_down=keep_head_down, 
            keep_head_straight=keep_head_straight, 
            look_down_init=look_down_init,
            search_pitch_explore=search_pitch_explore, 
            block_map_positions_if_fail=block_map_positions_if_fail,
            pix_T_camX=self.pix_T_camX,
            task=self.teach_task,
            depth_estimation_network=depth_network,
            )
        self.navigation.init_navigation(None)

        self.navigation.bring_head_to_angle(update_obs=False)
        
        origin_T_camX0 = utils.aithor.get_origin_T_camX(self.controller .last_event, False)

        # if args.use_attribute_detector and not args.use_gt_attributes and attribute_detector is None:
        #     from nets.attribute_detector import AttributeDetectorVLM as AttributeDetector
        #     self.attribute_detector = AttributeDetector(self.W, self.H, self.attributes)
        # else:
        #     self.attribute_detector = attribute_detector

        self.object_tracker = ObjectTrack(
            self.name_to_id, 
            self.id_to_name, 
            self.include_classes, 
            self.W, self.H, 
            pix_T_camX=self.pix_T_camX, 
            origin_T_camX0=origin_T_camX0, 
            ddetr=segmentation_network,
            controller=self.controller, 
            use_gt_objecttrack=use_gt_objecttrack,
            do_masks=True,
            id_to_mapped_id=self.id_to_mapped_id,
            navigation=self.navigation,
            use_mask_rcnn_pred=self.use_mask_rcnn_pred,
            use_open_set_segmenter=False,
            name_to_parsed_name=None,
            attribute_detector=attribute_detector,
            )
        self.teach_task.object_tracker = self.object_tracker

        if object_cat_pickup is not None and in_hand:
            # add holding object
            self.object_tracker.objects_track_dict[self.object_tracker.id_index] = {}
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['scores'] = 1.01
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['label'] = object_cat_pickup
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['locs'] = None
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['holding'] = True
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['can_use'] = True
            self.object_tracker.objects_track_dict[self.object_tracker.id_index]['sliced'] = False
            self.object_tracker.id_index += 1
        
        if args.create_movie and self.render:
            self.vis = Animation(
                self.W, self.H, 
                navigation=None if args.remove_map_vis else self.navigation, 
                name_to_id=self.name_to_id
                )
            self.movie_dir = os.path.join(self.output_folder, 'logging', self.folder_tag, f"movies")
            os.makedirs(self.movie_dir, exist_ok=True)
            # os.makedirs(args.movie_dir, exist_ok=True)
        else:
            self.vis = None

    def load_edh_instance(self, edh_instance):
        self.edh_instance: dict = None

        if edh_instance is None:
            logging.info("No EDH instance specified, defaulting to FloorPlan12_physics of world type kitchen")
            if self.er is None:
                self.er: EpisodeReplay = EpisodeReplay("thor", web_window_size=480)
            self.er.simulator.start_new_episode(world="FloorPlan12_physics", world_type="kitchen")
            return
        else:
            if self.er is None:
                self.er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])
            instance = load_json(edh_instance)
            check_task = InferenceRunner._get_check_task(instance, self.runner_config)
            game_file = InferenceRunner.get_game_file(instance, self.runner_config)
            instance_id = InferenceRunner._get_instance_id(edh_instance, instance)
            try:
                init_success, self.er = with_retry(
                    fn=lambda: InferenceRunner._initialize_episode_replay(instance, game_file, check_task,
                                                            self.runner_config.replay_timeout, self.er),
                    retries=self.runner_config.max_init_tries - 1,
                    check_first_return_value=True,
                )
                history_load_success, history_images = InferenceRunner._maybe_load_history_images(instance, self.runner_config)
                init_success = init_success and history_load_success
            except Exception:
                init_success = False
                print(f"Failed to initialize episode replay for instance={instance_id}")

            self.edh_instance = instance
            self.instance_id = instance_id
            self.game_id = self.edh_instance['game_id']
            self.metrics = create_new_traj_metrics(instance_id, self.game_id)
            self.metrics["init_success"] = init_success
            if args.use_progress_check:
                self.metrics_before_feedback = copy.deepcopy(self.metrics)
                self.metrics_before_feedback2 = copy.deepcopy(self.metrics)

            if not init_success:
                return init_success

            if "expected_init_goal_conditions_total" in instance and "expected_init_goal_conditions_satisfied" in instance:
                self.init_gc_total = instance["expected_init_goal_conditions_total"]
                self.init_gc_satisfied = instance["expected_init_goal_conditions_satisfied"]
            else:
                # For TfD instances, goal conditions are not cached so need an initial check
                (
                    _,
                    self.init_gc_total,
                    self.init_gc_satisfied,
                ) = InferenceRunner._check_episode_progress(self.er, check_task)

        self.er.simulator.is_record_mode = True
        return init_success

    def progress_check(self):
        '''
        User feedback
        '''
        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        progress_check_output = check_task.check_episode_progress(self.er.simulator.get_objects(self.controller.last_event), self.er.simulator)
        task_dict = {'dialog_history_cleaned':[], 'success':progress_check_output['success']}
        if not progress_check_output['success']:
            for subgoal in progress_check_output['subgoals']:
                if not subgoal['success']:
                    subgoal_description = subgoal['description']
                    subgoal_step_description = []
                    objects_mentioned = []
                    idx_to_phrase = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"]
                    for step in subgoal['steps']:
                        if not step['success']:
                            if step['objectId'] not in objects_mentioned:
                                objects_mentioned.append(step['objectId'])
                            object_idx = objects_mentioned.index(step['objectId'])
                            subgoal_step_description.append(f"For the {idx_to_phrase[object_idx]} {step['objectType']}: {step['desc']}")
                    subgoal_failed_text = f"You failed to complete the subtask: {subgoal['description']} "
                    for step_desc in subgoal_step_description:
                        subgoal_failed_text += f'{step_desc} '
                    task_dict['dialog_history_cleaned'].append(['Commander', subgoal_failed_text])
        return task_dict

    def get_goal_condition_success(self):
        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = InferenceRunner._check_episode_progress(self.er, check_task)
        return final_goal_conditions_satisfied, final_goal_conditions_total

    def eval(self, additional_tag=''):
        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = InferenceRunner._check_episode_progress(self.er, check_task)

        metrics_diff = evaluate_traj(
            success,
            self.edh_instance,
            self.teach_task.steps,
            self.init_gc_total,
            self.init_gc_satisfied,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        )
        self.teach_task.metrics.update(metrics_diff)

        progress_check_output = check_task.check_episode_progress(self.er.simulator.get_objects(self.controller.last_event), self.er.simulator)

        satisfied_objects = []
        if progress_check_output['satisfied_objects'] is not None:
            for idx in range(len(progress_check_output['satisfied_objects'])):
                satisfied_objects.append(progress_check_output['satisfied_objects'][idx]['objectType'])

        candidate_objects = []
        if progress_check_output['candidate_objects'] is not None:
            for idx in range(len(progress_check_output['candidate_objects'])):
                candidate_objects.append(progress_check_output['candidate_objects'][idx]['objectType'])

        failure_subgoal_description = []
        successful_subgoal_description = []
        failure_step_description = []
        successful_step_description = []
        for idx in range(len(progress_check_output['subgoals'])):
            subgoal = progress_check_output['subgoals'][idx]
            if not subgoal['success']:
                failure_subgoal_description.append(subgoal['description'])
            else:
                successful_subgoal_description.append(subgoal['description'])
            for step_idx in range(len(subgoal['steps'])):
                step = subgoal['steps'][step_idx]
                if not step['success']:
                    failure_step_description.append(step["desc"])
                else:
                    successful_step_description.append(step["desc"])

        print(f"Failed steps: {failure_step_description}")
        
        task_log = {
            "satisfied objects":str(satisfied_objects),
            "candidate objects":str(candidate_objects),
            "failed subtasks":str(failure_subgoal_description),
            "successful subtasks":str(successful_subgoal_description),
            "failed steps":str(failure_step_description),
            "successful steps":str(successful_step_description),
            }
        self.teach_task.metrics.update(task_log)

        cols = ["file"]+list(self.teach_task.metrics.keys())
        if 'pred_actions' in cols:
            cols.remove('pred_actions')
        tbl = wandb.Table(columns=cols)
        to_add_tbl = [self.tag]
        for k in list(self.teach_task.metrics.keys()):
            if k=="pred_actions":
                continue
            to_add_tbl.append(self.teach_task.metrics[k])
        tbl.add_data(*to_add_tbl)
        wandb.log({f"Metrics/{self.tag}": tbl})

        print("Eval:")
        print(additional_tag)
        print(f"Task success: {success}")
        print(f"Final subgoal success: {final_goal_conditions_satisfied} / {final_goal_conditions_total}")

        if self.log:
            # text_to_output = f'TAG: {self.folder_tag}\n\nPROMPT:\n\n{prompt}\n\n\n\nLLM CRITIC\n\n{decision}\n\nPLAN:\n{plan}\n\nSUMMARY:\n{skill_summary}\n\nPYTHON PROGRAM:\n\n{skill_function}'

            metrics_file = os.path.join(self.output_folder, 'logging', self.folder_tag, f"eval_metrics_success={success}.txt") #f'output/skill_logging'
            # os.makedirs(metrics_file, exist_ok=True)
            metrics_copy = copy.deepcopy(self.teach_task.metrics)
            del metrics_copy["pred_actions"]
            json_text = json.dumps(metrics_copy, indent=4)
            try:
                json_text += f'\n\n\n\n{self.initial_state}\n\n\n\n{self.executable_code}'
            except:
                pass
            
            # with open(metrics_file, 'w') as f:
            #     f.write(json_text.encode('utf-8'))

    def get_subgoal_success(self):
        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = InferenceRunner._check_episode_progress(self.er, check_task)
        return success, final_goal_conditions_total, final_goal_conditions_satisfied

    def run_tfd(self, user_progress_check=True):
        with open('prompt/run_script_template.txt') as f:
            python_script_template = f.read()
        if self.episode_in_try_except:
            try:
                self.search_dict = {}
                camX0_T_camXs = self.map_and_explore()
                print("RUNNING IN TRY EXCEPT")
                executable_code = self.run_llm(self.edh_instance)
                if args.run_from_code_file:
                    with open('output/executable_code_file.py') as f:
                        executable_code = f.read()
                # exec(self.skill_functions)
                executable_code_ = executable_code
                code_finished = ''
                for code_exec_i in range(args.num_allowable_failure):
                    replan = False
                    try:
                        executable_code = re.sub(r'(InteractionObject\()', r'\1self, ', executable_code_)
                        executable_code = re.sub(r'(AgentCorrective\()', r'\1self, ', executable_code)
                        self.executable_code = executable_code
                        exec(executable_code)

                    except CustomError as e:
                        replan = True
                        print(traceback.format_exc())
                        execution_error = str(traceback.format_exc())
                        failed_line = int(execution_error.split('exec(executable_code)\n  File "<string>", ')[1].split(',')[0].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ','')) - 1
                        # failed_line = int(execution_error.split(', in python_script')[0][-6:].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ',''))-4
                        failed_code = executable_code_.split('\n')[failed_line]
                        code_finished += '\n'.join(executable_code_.split('\n')[:failed_line])
                        code_remaining = '\n'.join(executable_code_.split('\n')[failed_line:])
                        only_error = execution_error.split(', in <module>\n')[-1]
                        execution_error = f"Code failed when executing line {failed_code} in the Python Script. {only_error}"
                        print(f"Execution Error: {execution_error}")
                        print(f"Failed code:\n{failed_code}")
                        print(f"Code already run:\n{code_finished}")
                        if code_exec_i==args.num_allowable_failure-1:
                            tbl = wandb.Table(columns=["execution error", "code finished", "code remaining", "revised code"])
                            tbl.add_data(execution_error, code_finished, code_remaining, "N/A")
                            wandb.log({f"LLM_error/replanning_{code_exec_i}_{self.tag}": tbl})
                            if self.log:
                                text_to_log = f"EXECUTION ERROR:\n{execution_error}\n\nCODE FINISHED:\n{code_finished}\n\nCODE REMAINING:\n{code_remaining}\n\nREVISED CODE:\nN\A"
                                os.makedirs(os.path.join(self.output_folder, 'logging', self.folder_tag, 'errors'), exist_ok=True)
                                with open(os.path.join(self.output_folder, 'logging', self.folder_tag, 'errors', f'error_{code_exec_i}.txt'), 'wb') as f:
                                    f.write(text_to_log.encode('utf-8'))
                            break
                        executable_code_ = self.run_llm_replan(
                                execution_error, 
                                code_remaining,
                                code_finished,
                                )
                        tbl = wandb.Table(columns=["execution error", "code finished", "code remaining", "revised code"])
                        tbl.add_data(execution_error, code_finished, code_remaining, executable_code_)
                        wandb.log({f"LLM_error/replanning_{code_exec_i}_{self.tag}": tbl})
                        if self.log:
                            text_to_log = f"EXECUTION ERROR:\n{execution_error}\n\nCODE FINISHED:\n{code_finished}\n\nCODE REMAINING:\n{code_remaining}\n\nREVISED CODE:\n{executable_code_}"
                            os.makedirs(os.path.join(self.output_folder, 'logging', self.folder_tag, 'errors'), exist_ok=True)
                            with open(os.path.join(self.output_folder, 'logging', self.folder_tag, 'errors', f'error_{code_exec_i}.txt'), 'wb') as f:
                                f.write(text_to_log.encode('utf-8'))
                    except Exception as e:
                        replan = True
                        print(traceback.format_exc())
                        execution_error = str(traceback.format_exc())
                        # failed_line = int(execution_error.split(', in python_script')[0][-6:].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ',''))-4
                        failed_line = int(execution_error.split('exec(executable_code)\n  File "<string>", ')[1].split(',')[0].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ','')) - 1
                        failed_code = executable_code_.split('\n')[failed_line]
                        code_finished += '\n'.join(executable_code_.split('\n')[:failed_line])
                        code_remaining = '\n'.join(executable_code_.split('\n')[failed_line:])
                        only_error = execution_error.split(', in <module>\n')[-1]
                        execution_error = f"Code failed when executing line {failed_code} in the Python Script: {only_error}"
                        print(f"Execution Error: {execution_error}")
                        print(f"Failed code:\n{failed_code}")
                        print(f"Code already run:\n{code_finished}")
                        if code_exec_i==args.num_allowable_failure-1:
                            tbl = wandb.Table(columns=["execution error", "code finished", "code remaining", "revised code"])
                            tbl.add_data(execution_error, code_finished, code_remaining, "N/A")
                            wandb.log({f"LLM_error/replanning_{code_exec_i}_{self.tag}": tbl})
                            if self.log:
                                text_to_log = f"EXECUTION ERROR:\n{execution_error}\n\nCODE FINISHED:\n{code_finished}\n\nCODE REMAINING:\n{code_remaining}\n\nREVISED CODE:\nN\A"
                                os.makedirs(os.path.join(self.output_folder, 'logging', self.folder_tag, 'errors'), exist_ok=True)
                                with open(os.path.join(self.output_folder, 'logging', self.folder_tag, 'errors', f'error_{code_exec_i}.txt'), 'wb') as f:
                                    f.write(text_to_log.encode('utf-8'))
                            break
                        executable_code_ = self.run_llm_replan(
                                execution_error, 
                                code_remaining,
                                code_finished,
                                )
                        tbl = wandb.Table(columns=["execution error", "code finished", "code remaining", "revised code"])
                        tbl.add_data(execution_error, code_finished, code_remaining, executable_code_)
                        wandb.log({f"LLM_error/replanning_{code_exec_i}_{self.tag}": tbl})
                        if self.log:
                            text_to_log = f"EXECUTION ERROR:\n{execution_error}\n\nCODE FINISHED:\n{code_finished}\n\nCODE REMAINING:\n{code_remaining}\n\nREVISED CODE:\n{executable_code_}"
                            os.makedirs(os.path.join(self.output_folder, 'logging', self.folder_tag, 'errors'), exist_ok=True)
                            with open(os.path.join(self.output_folder, 'logging', self.folder_tag, 'errors', f'error_{code_exec_i}.txt'), 'wb') as f:
                                f.write(text_to_log.encode('utf-8'))
                            
                    if not replan:
                        break

            except KeyboardInterrupt:
                # sys.exit(0)
                pass
            except Exception as e:
                tbl = wandb.Table(columns=["Error", "Traceback"])
                tbl.add_data(str(e), str(traceback.format_exc()))
                wandb.log({f"Errors/{self.tag}": tbl})
                print(e)
                print(traceback.format_exc())
        else:
            self.search_dict = {}
            camX0_T_camXs = self.map_and_explore()
            print("RUNNING IN TRY EXCEPT")
            executable_code = self.run_llm(self.edh_instance)
            if args.run_from_code_file:
                with open('output/executable_code_file.py') as f:
                    executable_code = f.read()
            # exec(self.skill_functions)
            executable_code_ = executable_code
            code_finished = ''
            for code_exec_i in range(args.num_allowable_failure):
                replan = False
                try:
                    executable_code = re.sub(r'(InteractionObject\()', r'\1self, ', executable_code_)
                    executable_code = re.sub(r'(AgentCorrective\()', r'\1self, ', executable_code)
                    self.executable_code = executable_code
                    exec(executable_code)

                except CustomError as e:
                    if code_exec_i==args.num_allowable_failure-1:
                        break
                    replan = True
                    print(traceback.format_exc())
                    execution_error = str(traceback.format_exc())
                    failed_line = int(execution_error.split('exec(executable_code)\n  File "<string>", ')[1].split(',')[0].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ','')) - 1
                    # failed_line = int(execution_error.split(', in python_script')[0][-6:].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ',''))-4
                    failed_code = executable_code_.split('\n')[failed_line]
                    code_finished += '\n'.join(executable_code_.split('\n')[:failed_line])
                    code_remaining = '\n'.join(executable_code_.split('\n')[failed_line:])
                    only_error = execution_error.split(', in <module>\n')[-1]
                    execution_error = f"Code failed when executing line {failed_code} in the Python Script. {only_error}"
                    print(f"Execution Error: {execution_error}")
                    print(f"Failed code:\n{failed_code}")
                    print(f"Code already run:\n{code_finished}")
                    executable_code_ = self.run_llm_replan(
                            execution_error, 
                            code_remaining,
                            code_finished,
                            )
                    tbl = wandb.Table(columns=["execution error", "code finished", "code remaining", "revised code"])
                    tbl.add_data(execution_error, code_finished, code_remaining, executable_code_)
                    wandb.log({f"LLM/{self.tag}_replanning_{code_exec_i}": tbl})
                except Exception as e:
                    if code_exec_i==args.num_allowable_failure-1:
                        break
                    replan = True
                    print(traceback.format_exc())
                    execution_error = str(traceback.format_exc())
                    # failed_line = int(execution_error.split(', in python_script')[0][-6:].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ',''))-4
                    failed_line = int(execution_error.split('exec(executable_code)\n  File "<string>", ')[1].split(',')[0].replace('l','').replace('i', '').replace('n','').replace('e','').replace(' ','')) - 1
                    failed_code = executable_code_.split('\n')[failed_line]
                    code_finished += '\n'.join(executable_code_.split('\n')[:failed_line])
                    code_remaining = '\n'.join(executable_code_.split('\n')[failed_line:])
                    only_error = execution_error.split(', in <module>\n')[-1]
                    execution_error = f"Code failed when executing line {failed_code} in the Python Script: {only_error}"
                    print(f"Execution Error: {execution_error}")
                    print(f"Failed code:\n{failed_code}")
                    print(f"Code already run:\n{code_finished}")
                    executable_code_ = self.run_llm_replan(
                            execution_error, 
                            code_remaining,
                            code_finished,
                            )
                    tbl = wandb.Table(columns=["execution error", "code finished", "code remaining", "revised code"])
                    tbl.add_data(execution_error, code_finished, code_remaining, executable_code_)
                    wandb.log({f"LLM/{self.tag}_replanning_{code_exec_i}": tbl})
                        
                if not replan:
                    break

    def run(self):
        if args.mode=="teach_eval_edh":
            self.run_edh()
        else:
            self.run_tfd(user_progress_check=self.use_progress_check)
        self.teach_task.step("Stop", None)
        self.eval()
        if self.controller is not None:
            self.controller.stop()
        self.render_output()

        return self.teach_task.metrics, self.er

    def render_output(self):
        if self.render:
            print("Rendering!")
            self.vis.render_movie(self.movie_dir, self.tag, tag=f"Full")
            frames_numpy = np.asarray(self.vis.image_plots)
            frames_numpy = np.transpose(frames_numpy, (0,3,1,2))
            wandb.log({f"movies/{self.tag}": wandb.Video(frames_numpy, fps=10, format="mp4")})

    # Adds Navigate subgoal between two consecutive Object Interaction Subgoals
    def add_navigation_goals(self, subgoals, objects):
        final_subgoals = copy.deepcopy(subgoals)
        final_objects = copy.deepcopy(objects)
        obj_interaction = False
        idx_add = 0 
        for i in range(len(subgoals)):
            if subgoals[i]!="Navigate" and subgoals[i-1]!="Navigate":
                final_subgoals.insert((i-1)+idx_add, "Navigate")
                final_objects.insert((i-1)+idx_add, objects[i])
                idx_add += 1
        return final_subgoals, final_objects

    def get_image(self, controller=None):
        if controller is not None:
            rgb = controller.last_event.frame
        else:
            raise NotImplementedError
        return rgb

def run_teach():
    save_metrics = True
    split_ = args.split
    data_dir = args.teach_data_dir 
    output_dir = "./plots/subgoal_output"
    images_dir = "./plots/subgoal_output"
    if args.mode=="teach_eval_tfd":
        instance_dir = os.path.join(data_dir, f"tfd_instances/{split_}")
    elif args.mode=="teach_eval_edh":
        instance_dir = os.path.join(data_dir, f"edh_instances/{split_}")
    files = os.listdir(instance_dir) # sample every other

    if not os.path.exists(f'./data/sorted_task_files_{split_}.json'):
        from tqdm import tqdm
        sorted_task_files_objects = {}
        sorted_task_files = {}
        task_name_to_descs = {}
        runner_config = InferenceRunnerConfig(
            data_dir=data_dir,
            split=split_,
            output_dir=output_dir,
            images_dir=images_dir,
            model_class=None,
            model_args=None,
            num_processes=1,
            max_init_tries=5,
            replay_timeout=500,
            max_api_fails=args.max_api_fails,
            max_traj_steps=args.max_traj_steps,
        )
        for file in tqdm(files):
            task_instance = os.path.join(instance_dir, file)
            instance = load_json(task_instance)
            game_file = InferenceRunner.get_game_file(instance, runner_config)
            game = load_json(game_file)
            task_description = game["tasks"][0]['task_name'].replace('.', '').replace(' ', '_') #game["tasks"][0]['task_name'].replace('.', '').replace(' ', '_') #game["tasks"][0]['desc'].replace('.', '').replace(' ', '_')
            task_description_object = game["tasks"][0]['desc'].replace('.', '').replace(' ', '_')
            if task_description not in sorted_task_files.keys():
                sorted_task_files[task_description] = []
                task_name_to_descs[task_description] = set()
            sorted_task_files[task_description].append(file)
            task_name_to_descs[task_description].add(task_description_object)
            if task_description_object not in sorted_task_files_objects.keys():
                sorted_task_files_objects[task_description_object] = []
            sorted_task_files_objects[task_description_object].append(file)
        save_dict_as_json(sorted_task_files, f'./data/sorted_task_files_{split_}.json')
        save_dict_as_json(sorted_task_files_objects, f'./data/sorted_task_files_{split_}_taskparams.json')
        task_name_to_descs = {k:list(v) for k,v in task_name_to_descs.items()}
        save_dict_as_json(task_name_to_descs, f'./data/task_name_to_descs_{split_}.json')

    if args.sample_every_other:
        files = files[::2]

    if args.episode_file is not None:
        files_idx = files.index(args.episode_file)
        files = files[files_idx:files_idx+1]

    if args.max_episodes is not None:
        files = files[:args.max_episodes]

    # initialize wandb
    if args.set_name=="test00":
        wandb.init(mode="disabled")
    else:
        wandb.init(project="embodied-llm-memory-learning", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)

    metrics = {}
    metrics_file = os.path.join(args.metrics_dir, f'{args.mode}_metrics_{split_}.txt')
    if os.path.exists(metrics_file) and args.skip_if_exists:
        metrics = load_json(metrics_file)
    if args.use_progress_check:
        metrics_before_feedback = {}
        metrics_before_feedback2 = {}
        metrics_file_before_feedback = os.path.join(args.metrics_dir, f'{args.mode}_metrics_before_feedback_{split_}.txt')
        metrics_file_before_feedback2 = os.path.join(args.metrics_dir, f'{args.mode}_metrics_before_feedback2_{split_}.txt')
        if os.path.exists(metrics_file_before_feedback) and args.skip_if_exists:
            metrics_before_feedback = load_json(metrics_file_before_feedback)
        if os.path.exists(metrics_file_before_feedback2) and args.skip_if_exists and args.use_progress_check:
            metrics_before_feedback2 = load_json(metrics_file_before_feedback2)
    iter_ = 0
    er = None
    depth_estimation_network = None
    segmentation_network = None
    attribute_detector = None
    clip = None
    for file in files:
        print("Running ", file)
        print(f"Iteration {iter_+1}/{len(files)}")
        if args.skip_if_exists and (file in metrics.keys()):
            print(f"File already in metrics... skipping...")
            iter_ += 1
            continue
        task_instance = os.path.join(instance_dir, file)
        subgoalcontroller = SubGoalController(
                data_dir, 
                output_dir, 
                images_dir, 
                task_instance, 
                iteration=iter_, 
                er=er, 
                depth_network=depth_estimation_network, 
                segmentation_network=segmentation_network,
                attribute_detector=attribute_detector,
                clip=clip,
                )
        if subgoalcontroller.init_success:
            metrics_instance, er = subgoalcontroller.run()
            if segmentation_network is None:
                segmentation_network = subgoalcontroller.object_tracker.ddetr
            if depth_estimation_network is None:
                depth_estimation_network = subgoalcontroller.navigation.depth_estimator
            if attribute_detector is None:
                attribute_detector = subgoalcontroller.object_tracker.attribute_detector
            if clip is None:
                clip = subgoalcontroller.clip
        else:
            metrics_instance, er = subgoalcontroller.teach_task.metrics, subgoalcontroller.er
        metrics[file] = metrics_instance
        if args.use_progress_check:
            metrics_instance_before_feedback = subgoalcontroller.metrics_before_feedback
            metrics_before_feedback[file] = metrics_instance_before_feedback
            metrics_instance_before_feedback2 = subgoalcontroller.metrics_before_feedback2
            metrics_before_feedback2[file] = metrics_instance_before_feedback2

        iter_ += 1

        if save_metrics:
            from teach.eval.compute_metrics import aggregate_metrics

            aggregrated_metrics = aggregate_metrics(metrics, args)

            print('\n\n---------- File 1 ---------------')
            to_log = []  
            to_log.append('-'*40 + '-'*40)
            list_of_files = files #list(metrics.keys())
            # to_log.append(f'Files: {str(list_of_files)}')
            to_log.append(f'Split: {split_}')
            to_log.append(f'Number of files: {len(list(metrics.keys()))}')
            for f_n in aggregrated_metrics.keys(): #keys_include:
                to_log.append(f'{f_n}: {aggregrated_metrics[f_n]}') 
            to_log.append('-'*40 + '-'*40)

            os.makedirs(args.metrics_dir, exist_ok=True)
            path = os.path.join(args.metrics_dir, f'{args.mode}_summary_{split_}.txt')
            with open(path, "w") as fobj:
                for x in to_log:
                    fobj.write(x + "\n")

            save_dict_as_json(metrics, metrics_file)

            aggregrated_metrics["num episodes"] = iter_
            tbl = wandb.Table(columns=list(aggregrated_metrics.keys()))
            tbl.add_data(*list(aggregrated_metrics.values()))
            wandb.log({f"Metrics_summary/Summary": tbl, 'step':iter_})                

            cols = ["file"]+list(metrics_instance.keys())
            cols.remove('pred_actions')
            tbl = wandb.Table(columns=cols)
            for f_k in metrics.keys():
                to_add_tbl = [f_k]
                for k in list(metrics[f_k].keys()):
                    if k=="pred_actions":
                        continue
                    to_add_tbl.append(metrics[f_k][k])
                # list_values = [f_k] + list(metrics[f_k].values())
                tbl.add_data(*to_add_tbl)
            wandb.log({f"Metrics_summary/Metrics": tbl, 'step':iter_})
