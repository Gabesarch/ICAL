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
import random

import numpy as np
import os
import glob
import cv2

import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torchvision.transforms as T

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
if args.mode in ["teach_eval_tfd", "teach_eval_custom", "teach_eval_continual", "teach_skill_learning"]:
    from teach.inference.tfd_inference_runner import TfdInferenceRunner as InferenceRunner
elif args.mode=="teach_eval_edh":
    from teach.inference.edh_inference_runner import EdhInferenceRunner as InferenceRunner
else:
    assert(False) # what mode is this? 
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

import openai

logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
                filename='./subgoalcontroller.log',
                filemode='w'
            )

from IPython.core.debugger import set_trace
from PIL import Image
import wandb
import re
from tqdm import tqdm

from .teach_eval_embodied_llm import SubGoalController
from .agent.api_primitives_executable import InteractionObject, CustomError
from deepdiff import DeepDiff
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

class ProgramGen:
    def __init__(self, **kwargs):
        '''
        Functions for simulating, interacting, and refinement
        '''
        pass

    def get_expert_program(
        self,
        return_states=False
    ):
        '''
        Gets Python program of expert demonstration and initial states
        '''
        attributes = set([
            'isToggled', 
            'isFilledWithLiquid', 
            'fillLiquid', 
            'isDirty', 
            'isUsedUp', 
            'isCooked', 
            'isSliced', 
            'isOpen', 
            'isPickedUp'
            ]) # 'temperature',
        attributes_check = {
            'isToggled':'toggleable', 
            'isFilledWithLiquid':'canFillWithLiquid', 
            'fillLiquid':'canFillWithLiquid', 
            'isDirty':'dirtyable',
            'isUsedUp':'canBeUsedUp',
            'isCooked':'cookable',
            'isSliced':'sliceable',
            'isOpen':'openable',
            'isPickedUp':'pickupable',
            }

        self.navigation.bring_head_to_angle(update_obs=True, angle=30)

        self.origin_objects = self.controller.last_event.metadata["objects"]
        object_dict = {obj["objectId"]:obj for obj in self.origin_objects}
        relevant_objects = {}
        object_number = {}
        number = 1
        for action in self.edh_instance['driver_actions_future']:
            if action["action_name"] in ["Place", "Pickup", 'Open', 'Close', "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                if action['oid'] not in relevant_objects.keys() and action['oid'] in object_dict.keys():
                    relevant_objects[action['oid']] = object_dict[action['oid']]
                    object_number[action['oid']] = number
                    number += 1
        relevant_objects_initial = copy.deepcopy(relevant_objects)

        self.object_tracker.update(
            np.zeros((3, self.W, self.H)),
            np.zeros((self.W, self.H)),
            None,
            )

        initial_state = self.get_current_state(self.object_tracker.objects_track_dict, metadata=self.controller.last_event.metadata)
        initial_state_object_track = copy.deepcopy(self.object_tracker.objects_track_dict)
        metadata_before_execution = copy.deepcopy(self.controller.last_event.metadata)
        metaid_to_key = {}
        for k in initial_state_object_track.keys():
            metaid_to_key[initial_state_object_track[k]['metaID']] = k
        max_id = max(list(metaid_to_key.values()))

        subgoals = []
        object_ids_relevant = set()
        for action in tqdm(self.edh_instance['driver_actions_future'], leave=False):
            if action["action_name"] in ["Text", "Navigation", "SearchObject", "SelectOid", "OpenProgressCheck"]:
                continue
            if 'x' in action.keys():
                obj_relative_coord = [action['y'], action['x']]
            else:
                obj_relative_coord = None
            sim_succ, err_message, help_message = InferenceRunner._execute_action(self.er.simulator, action["action_name"], obj_relative_coord)
            if not sim_succ:
                print(err_message, help_message)
                print("Action failed during replay.. skipping this episode..")
                return 
            current_objects = self.controller.last_event.metadata["objects"]
            current_object_dict = {obj["objectId"]:obj for obj in current_objects}
            if action['oid'] is not None and action['oid'] not in relevant_objects.keys():
                relevant_objects[action['oid']] = current_object_dict[action['oid']]
                relevant_objects_initial[action['oid']] = current_object_dict[action['oid']]
                object_number[action['oid']] = number
                number += 1
            if action["action_name"] in ["Place", "Pickup", 'Open', 'Close', "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                object_name = current_object_dict[action['oid']]["objectType"] #action['oid'].split('|')[0]
                if "Sliced" in action['oid'] and "Sliced" not in object_name:
                    object_name += "Sliced"
                if action["oid"] not in metaid_to_key.keys():
                    object_name += f'_{max_id+1}'
                    metaid_to_key[action["oid"]] = max_id+1
                    max_id += 1
                else:
                    object_name += f'_{metaid_to_key[action["oid"]]}'
                subgoals.extend([["Navigate", object_name], [action["action_name"], object_name]])
                object_ids_relevant.add(int(object_name.split('_')[-1]))
                interacted_object = relevant_objects[action['oid']]
                for k in relevant_objects.keys():
                    cur_state = current_object_dict[k]
                    prev_state = relevant_objects[k]
                    object_cur_type = cur_state["objectType"]
                    if "Sliced" in cur_state['objectId'] and "Sliced" not in object_cur_type:
                        object_cur_type += "Sliced"

                    if k not in metaid_to_key.keys():
                        object_cur_type += f'_{max_id+1}'
                        metaid_to_key[k] = max_id+1
                        max_id += 1
                    else:
                        object_cur_type += f'_{metaid_to_key[k]}'
                    prev_state = {k:v for k,v in prev_state.items() if k in attributes}
                    cur_state = {k:v for k,v in cur_state.items() if k in attributes}
                    diff = list(set(cur_state.items()) ^ set(prev_state.items()))
                    diff_keys = list(set([l[0] for l in diff]))
                    if len(diff)>0:
                        for diff_k in diff_keys:
                            state_val = cur_state[diff_k]
                            if type(state_val)==str:
                                state_val = f'"{state_val}"'
                            subgoals.extend([["StateChange", (object_cur_type, self.object_tracker.metadata_to_attributes[diff_k], state_val)]])
                            object_ids_relevant.add(int(object_cur_type.split('_')[-1]))
                        relevant_objects[k] = current_object_dict[k]

        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        progress_check_output = check_task.check_episode_progress(self.er.simulator.get_objects(self.er.simulator.controller.last_event), self.er.simulator)
        description = progress_check_output["description"].replace('.', '').replace(' ', '_')
        text = '' #'-----DIALOGUE----\n'
        for line in self.edh_instance['dialog_history_cleaned']:
            text += f'<{line[0]}> {line[1]}\n'
        self.llm.search_dict = {}

        state_before_object_track = {k:v for k,v in initial_state_object_track.items() if k in object_ids_relevant}
        state_before = self.get_current_state(state_before_object_track, include_location=False, metadata=metadata_before_execution)

        program = self.llm.subgoals_to_program(subgoals, self.object_tracker.get_label_of_holding(), initial_states=initial_state_object_track)

        root = os.path.join(self.demo_folder, self.task_type) #f'output/expert_programs2/{description}'
        os.makedirs(root, exist_ok=True)

        program = f'Dialogue Instruction:\n{text}\nInitial Object State:\n{state_before}\nDemonstration Script:\n```python\n{program}```'
        with open(os.path.join(root, f'{self.tag.split(".")[0]}.txt'), 'w') as f:
            f.write(program)

    def get_expert_program_idm(
        self,
        return_states=False
    ):
        '''
        Gets Python program of expert demonstration and initial states
        '''
        attributes = set([
            'isToggled', 
            'isFilledWithLiquid', 
            'fillLiquid', 
            'isDirty', 
            'isUsedUp', 
            'isCooked', 
            'isSliced', 
            'isOpen', 
            'isPickedUp'
            ]) # 'temperature',
        attributes_check = {
            'isToggled':'toggleable', 
            'isFilledWithLiquid':'canFillWithLiquid', 
            'fillLiquid':'canFillWithLiquid', 
            'isDirty':'dirtyable',
            'isUsedUp':'canBeUsedUp',
            'isCooked':'cookable',
            'isSliced':'sliceable',
            'isOpen':'openable',
            'isPickedUp':'pickupable',
            }

        self.navigation.bring_head_to_angle(update_obs=True, angle=30)

        self.origin_objects = self.controller.last_event.metadata["objects"]
        object_dict = {obj["objectId"]:obj for obj in self.origin_objects}
        relevant_objects = {}
        object_number = {}
        number = 1
        for action in self.edh_instance['driver_actions_future']:
            if action["action_name"] in ["Place", "Pickup", 'Open', 'Close', "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                if action['oid'] not in relevant_objects.keys() and action['oid'] in object_dict.keys():
                    relevant_objects[action['oid']] = object_dict[action['oid']]
                    object_number[action['oid']] = number
                    number += 1
        relevant_objects_initial = copy.deepcopy(relevant_objects)

        self.object_tracker.update(
            np.zeros((3, self.W, self.H)),
            np.zeros((self.W, self.H)),
            None,
            )

        initial_state = self.get_current_state(self.object_tracker.objects_track_dict, metadata=self.controller.last_event.metadata)
        initial_state_object_track = copy.deepcopy(self.object_tracker.objects_track_dict)
        metadata_before_execution = copy.deepcopy(self.controller.last_event.metadata)
        metaid_to_key = {}
        for k in initial_state_object_track.keys():
            metaid_to_key[initial_state_object_track[k]['metaID']] = k
        max_id = max(list(metaid_to_key.values()))

        subgoals = []
        # object_ids_relevant = set()
        object_class_relevant = set()
        rgb = self.controller.last_event.frame #cv2.resize(self.controller.last_event.frame, dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
        images = [rgb]
        for action in tqdm(self.edh_instance['driver_actions_future'], leave=False):
            if action["action_name"] in ["Text", "Navigation", "SearchObject", "SelectOid", "OpenProgressCheck"]:
                rgb = self.controller.last_event.frame #cv2.resize(self.controller.last_event.frame, dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
                images.append(rgb)
                continue
            if 'x' in action.keys():
                obj_relative_coord = [action['y'], action['x']]
            else:
                obj_relative_coord = None
            sim_succ, err_message, help_message = InferenceRunner._execute_action(self.er.simulator, action["action_name"], obj_relative_coord)
            if not sim_succ:
                print(err_message, help_message)
                print("Action failed during replay.. skipping this episode..")
                return 
            current_objects = self.controller.last_event.metadata["objects"]
            current_object_dict = {obj["objectId"]:obj for obj in current_objects}
            if action['oid'] is not None and action['oid'] not in relevant_objects.keys():
                relevant_objects[action['oid']] = current_object_dict[action['oid']]
                relevant_objects_initial[action['oid']] = current_object_dict[action['oid']]
                object_number[action['oid']] = number
                number += 1
            rgb = self.controller.last_event.frame #cv2.resize(self.controller.last_event.frame, dsize=(480, 480), interpolation=cv2.INTER_CUBIC)
            images.append(rgb)
            if action["action_name"] in ["Pan Left", "Pan Right"]:
                continue
            predicted_action, predicted_label = self.run_idm(images[-2:])
            try:
                actual_obj = current_object_dict[action['oid']]['objectType']
            except:
                actual_obj = None
            if predicted_action not in ["Place", "Pickup", 'Open', 'Close', "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                predicted_label = None
            # action["action_name"] = predicted_action
            if (predicted_label!="no_object") and predicted_label and predicted_action and predicted_action in ["Place", "Pickup", 'Open', 'Close', "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                object_name = predicted_label
                if object_name in self.name_to_mapped_name.keys():
                    object_name = self.name_to_mapped_name[object_name]
                print(f"{predicted_action} {object_name} (actual: {action['action_name']} {actual_obj})", )
                subgoals.extend([["Navigate", object_name], [predicted_action, object_name]])
                object_class_relevant.add(object_name)

        check_task = InferenceRunner._get_check_task(self.edh_instance, self.runner_config)
        progress_check_output = check_task.check_episode_progress(self.er.simulator.get_objects(self.er.simulator.controller.last_event), self.er.simulator)
        description = progress_check_output["description"].replace('.', '').replace(' ', '_')
        text = '' #'-----DIALOGUE----\n'
        for line in self.edh_instance['dialog_history_cleaned']:
            text += f'<{line[0]}> {line[1]}\n'
        self.llm.search_dict = {}

        # state_before_object_track = {k:v for k,v in initial_state_object_track.items() if k in object_ids_relevant}
        state_before_object_track = {k:v for k,v in initial_state_object_track.items() if initial_state_object_track[k]["label"] in object_class_relevant}
        state_before = self.get_current_state(state_before_object_track, include_location=False, metadata=metadata_before_execution)
        program = self.llm.subgoals_to_program(subgoals, self.object_tracker.get_label_of_holding(), initial_states=initial_state_object_track)

        root = os.path.join(self.demo_folder, self.task_type)
        os.makedirs(root, exist_ok=True)

        program_ = f'Dialogue Instruction:\n{text}\nInitial Object State:\n{state_before}\nDemonstration Script:\n```python\n{program}```'
        with open(os.path.join(root, f'{self.tag.split(".")[0]}.txt'), 'w') as f:
            f.write(program_)

        root_all_objs = root.replace('task_demos', 'task_demos_all_objs')
        os.makedirs(root_all_objs, exist_ok=True)

        state_before_object_track = {k:v for k,v in initial_state_object_track.items()}
        state_before = self.get_current_state(state_before_object_track, include_location=False, metadata=metadata_before_execution)

        program_ = f'Dialogue Instruction:\n{text}\nInitial Object State:\n{state_before}\nDemonstration Script:\n```python\n{program}```'
        with open(os.path.join(root_all_objs, f'{self.tag.split(".")[0]}.txt'), 'w') as f:
            f.write(program_)

    
    def init_idm(self):
        from nets.inverse_dynamics_model import IDM
        self.initialize_constants()

        num_classes = len(self.include_classes) # minus one to remove no object

        load_pretrained = True
        self.idm = IDM(
            num_classes, 
            load_pretrained, 
            num_actions=len(self.actions2idx),
            actions2idx=self.actions2idx
            )
        self.idm.to(device)
        self.idm.eval()

        path = args.load_model_path
        _, _ = saverloader.load_from_path(
                path, 
                self.idm, 
                None, 
                strict=True, 
                lr_scheduler=None,
                )

    @torch.no_grad()
    def run_idm(self, images):
        if len(images)!=2:
            assert False # only accepts 2 frames

        image_mean = np.array([0.485,0.456,0.406]).reshape(1,3,1,1)
        image_std = np.array([0.229,0.224,0.225]).reshape(1,3,1,1)
        resize_transform = T.Resize((480,480))

        images_ = [np.asarray(resize_transform(Image.fromarray(image))) for image in images]
        images_ = np.asarray(images_) * 1./255
        images_ = images_.transpose(0, 3, 1, 2)
        images_ = (images_ - image_mean) / image_std
        images_ = images_.astype(np.float32)
        images_ = torch.from_numpy(images_).to(device).unsqueeze(0)

        # put on cuda
        # images = sample_batched["images"].to(device)
                        
        out_dict = self.idm.predict(
            images_, 
            )
        predicted_action = self.actions[out_dict["pred_action"]]
        # predicted_label = self.id_to_name[out_dict["pred_label"]]
        predicted_label = self.id_to_name[out_dict["pred_label_forceobject"]]

        return predicted_action, predicted_label

    def initialize_constants(self):
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
        
        alfred_objects, object_mapping  = get_alfred_constants()

        for obj in alfred_objects:
            if obj not in self.include_classes:
                self.include_classes.append(obj)

        self.special_classes = ['AppleSliced', 'BreadSliced', 'EggCracked', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']
        self.include_classes += self.special_classes
        
        self.include_classes += ["no_object"]

        actions = [
            'MoveAhead', 
            'RotateRight', 
            'RotateLeft', 
            'LookDown',
            'LookUp',
            'PickupObject', 
            'PutObject', 
            'OpenObject', 
            'CloseObject', 
            'SliceObject',
            'ToggleObjectOn',
            'ToggleObjectOff',
            # 'Done',
            ]

        actions = [
                'Forward',
                'Backward', 
                'Turn Right', 
                'Turn Left', 
                'Look Down',
                'Look Up',
                'Pan Left',
                'Pan Right',
                'Pickup', 
                'Place', 
                'Open', 
                'Close', 
                'Slice',
                'ToggleOn',
                'ToggleOff',
                'Pour',
                ]

        subgoals = [
            'PickupObject', 
            'PutObject', 
            'OpenObject', 
            'CloseObject', 
            'SliceObject',
            'GotoLocation',
            'HeatObject',
            "ToggleObject",
            "CleanObject",
            "HeatObject",
            "CoolObject",
            ]

        # self.action_mapping = {
        #     'MoveAhead':'Forward', 
        #     'RotateRight': 'Turn Right', 
        #     'RotateLeft': 'Turn Left', 
        #     'LookDown': 'Look Down',
        #     'LookUp': 'Look Up',
        #     'PickupObject': 'Pickup', 
        #     'PutObject': 'Place', 
        #     'OpenObject': 'Open', 
        #     'CloseObject': 'Close', 
        #     'SliceObject': 'Slice',
        #     'ToggleObjectOn': 'ToggleOn',
        #     'ToggleObjectOff': 'ToggleOff',
        #     'Pour': 'Pour',
        # }

        
        self.actions = {i:actions[i] for i in range(len(actions))}
        self.actions2idx = {actions[i]:i for i in range(len(actions))}
        self.subgoals2idx = {subgoals[i]:i for i in range(len(subgoals))}
        self.idx2subgoals = {i:subgoals[i] for i in range(len(subgoals))}

        self.name_to_id = {}
        self.id_to_name = {}
        self.instance_counter = {}
        idx = 0
        for name in self.include_classes:
            self.name_to_id[name] = idx
            self.id_to_name[idx] = name
            self.instance_counter[name] = 0
            idx += 1

def get_alfred_constants():

    OBJECTS = [
            'AlarmClock',
            'Apple',
            'ArmChair',
            'BaseballBat',
            'BasketBall',
            'Bathtub',
            'BathtubBasin',
            'Bed',
            'Blinds',
            'Book',
            'Boots',
            'Bowl',
            'Box',
            'Bread',
            'ButterKnife',
            'Cabinet',
            'Candle',
            'Cart',
            'CD',
            'CellPhone',
            'Chair',
            'Cloth',
            'CoffeeMachine',
            'CounterTop',
            'CreditCard',
            'Cup',
            'Curtains',
            'Desk',
            'DeskLamp',
            'DishSponge',
            'Drawer',
            'Dresser',
            'Egg',
            'FloorLamp',
            'Footstool',
            'Fork',
            'Fridge',
            'GarbageCan',
            'Glassbottle',
            'HandTowel',
            'HandTowelHolder',
            'HousePlant',
            'Kettle',
            'KeyChain',
            'Knife',
            'Ladle',
            'Laptop',
            'LaundryHamper',
            'LaundryHamperLid',
            'Lettuce',
            'LightSwitch',
            'Microwave',
            'Mirror',
            'Mug',
            'Newspaper',
            'Ottoman',
            'Painting',
            'Pan',
            'PaperTowel',
            'PaperTowelRoll',
            'Pen',
            'Pencil',
            'PepperShaker',
            'Pillow',
            'Plate',
            'Plunger',
            'Poster',
            'Pot',
            'Potato',
            'RemoteControl',
            'Safe',
            'SaltShaker',
            'ScrubBrush',
            'Shelf',
            'ShowerDoor',
            'ShowerGlass',
            'Sink',
            'SinkBasin',
            'SoapBar',
            'SoapBottle',
            'Sofa',
            'Spatula',
            'Spoon',
            'SprayBottle',
            'Statue',
            'StoveBurner',
            'StoveKnob',
            'DiningTable',
            'CoffeeTable',
            'SideTable',
            'TeddyBear',
            'Television',
            'TennisRacket',
            'TissueBox',
            'Toaster',
            'Toilet',
            'ToiletPaper',
            'ToiletPaperHanger',
            'ToiletPaperRoll',
            'Tomato',
            'Towel',
            'TowelHolder',
            'TVStand',
            'Vase',
            'Watch',
            'WateringCan',
            'Window',
            'WineBottle',
        ]

    # SLICED = [
    #     'AppleSliced',
    #     'BreadSliced',
    #     'LettuceSliced',
    #     'PotatoSliced',
    #     'TomatoSliced'
    # ]

    # OBJECTS += SLICED

    # object parents
    OBJ_PARENTS = {obj: obj for obj in OBJECTS}
    OBJ_PARENTS['AppleSliced'] = 'Apple'
    OBJ_PARENTS['BreadSliced'] = 'Bread'
    OBJ_PARENTS['LettuceSliced'] = 'Lettuce'
    OBJ_PARENTS['PotatoSliced'] = 'Potato'
    OBJ_PARENTS['TomatoSliced'] = 'Tomato'

    return OBJECTS, OBJ_PARENTS