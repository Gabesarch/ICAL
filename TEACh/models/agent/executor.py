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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
np.random.seed(args.seed)


class ExecuteController:
    def __init__(self):
        '''
        responsible for executing actions + subgoals
        inherited by models.SubGoalController
        '''
        pass

    def map_and_explore(self, render=False):
        self.object_tracker.check_if_centroid_falls_within_map = False
        max_explore_steps = args.explore_steps
        if args.load_explore:
            trial_name = self.tag #f"{self.edh_instance['instance_id']}_{self.edh_instance['game_id']}" 
            map_path = os.path.join(args.precompute_map_path, trial_name+'.p')
            if os.path.exists(map_path):
                print(f"Loading {map_path}...")
                with open(map_path, 'rb') as config_dictionary_file:
                    navigation_explore_state = pickle.load(config_dictionary_file)
                self.navigation.explorer.mapper = navigation_explore_state[0][0]
                self.navigation.explorer.obstructed_states = navigation_explore_state[0][1]
                self.navigation.explorer.step_count = navigation_explore_state[0][2]
                self.object_tracker.objects_track_dict = navigation_explore_state[1]
                self.object_tracker.id_index = max(list(self.object_tracker.objects_track_dict.keys()))
                self.navigation.obs.image_list = [np.zeros((3, self.W, self.H))]
                self.navigation.obs.depth_map_list = [np.zeros((self.W, self.H))]
                self.navigation.obs.return_status = "SUCCESSFUL"
                self.navigation.explorer.return_status = "SUCCESSFUL"
                self.navigation.explorer.prev_act_id = self.navigation.explorer.actions_inv['pass']
            else:
                self.navigation.explore_env(object_tracker=self.object_tracker, vis=self.vis, return_obs_dict=False, max_fail=5, max_steps=max_explore_steps)
                navigation_explore_state = [
                    [copy.deepcopy(self.navigation.explorer.mapper),
                     copy.deepcopy(self.navigation.explorer.obstructed_states),
                     copy.deepcopy(self.navigation.explorer.step_count)],
                    copy.deepcopy(self.object_tracker.objects_track_dict)
                    ]
                os.makedirs(args.precompute_map_path, exist_ok=True)
                print(f"Saving {map_path}...")
                with open(map_path, "wb") as file_:
                    pickle.dump(navigation_explore_state, file_, -1)
                if args.use_odin:
                    camX0_T_camX = self.navigation.explorer.get_camX0_T_camX()
                    rgb, depth = self.navigation.get_obs(head_tilt=self.navigation.explorer.head_tilt)
                    self.object_tracker.update(rgb, depth, camX0_T_camX, vis=self.vis, force_update_odin=True)
                if self.vis is not None:
                    for _ in range(5):
                        self.vis.add_frame(self.get_image(self.controller), text=f"FINAL")
                    # render movie of agent
                    # self.vis.render_movie(args.movie_dir, 0, tag=f"explore_{self.tag}")
        else:
            self.navigation.explore_env(object_tracker=self.object_tracker, vis=self.vis, return_obs_dict=False, max_fail=5, max_steps=max_explore_steps)
            if args.use_odin:
                camX0_T_camX = self.navigation.explorer.get_camX0_T_camX()
                rgb, depth = self.navigation.get_obs(head_tilt=self.navigation.explorer.head_tilt)
                self.object_tracker.update(rgb, depth, camX0_T_camX, vis=self.vis, force_update_odin=True)
            if self.vis is not None:
                for _ in range(5):
                    self.vis.add_frame(self.get_image(self.controller), text=f"FINAL")
                # render movie of agent
                # self.vis.render_movie(args.movie_dir, 0, tag=f"explore_{self.tag}")

        print("forcing closed walls in map...")
        self.navigation.explorer.mapper.force_closed_walls()

        print("Removing centroids outside map bounds...")
        self.object_tracker.filter_centroids_out_of_bounds()
        self.object_tracker.check_if_centroid_falls_within_map = True

        if self.vis is not None:
            for _ in range(10):
                self.vis.add_frame(self.get_image(self.controller), text="FINAL POS MAPPING")

    def random_search(self, target, max_steps=200):
        print("Searching for object!")
        if args.use_estimated_depth:
            self.navigation.bring_head_to_angle(angle=0, vis=self.vis)
            search_mode = True
        else:
            self.navigation.bring_head_to_angle(update_obs=True, vis=self.vis)
            search_mode = True
        out = self.navigation.search_random_locs_for_object(
                target,
                max_steps=max_steps, 
                vis=self.vis, 
                text=f'search for {target}', 
                object_tracker=self.object_tracker, 
                max_fail=30, 
                search_mode=search_mode,
                num_search_locs_object=args.num_search_locs_object,
                )
        return out

    def search_near_object(self, target, receptacle_target, max_rec_to_search=10, search_mode = False, max_steps=150, steps_cur=1):
        '''
        Search for target object near another receptacle_target object
        max_rec_to_search: if there are many of this receptacle in memory, how many (max) to search?
        '''

        if type(receptacle_target)==int:
            obj_ID = receptacle_target
            receptacle_target = self.object_tracker.objects_track_dict[obj_ID]['label']
            centroids_receptacle = np.asarray([self.object_tracker.objects_track_dict[obj_ID]['locs']])
            labels_receptacle = [receptacle_target]
            object_ids_receptacle = np.asarray([obj_ID])
        else:
            centroids_receptacle, labels_receptacle, object_ids_receptacle = self.object_tracker.get_centroids_and_labels(
                    return_ids=True, object_cat=receptacle_target
                    )
                    
        if len(centroids_receptacle)==0:
            print(f"None of {receptacle_target} in memory..")
            return {}

        starts_steps = self.teach_task.steps

        for c_idx in range(min(len(centroids_receptacle), max_rec_to_search)):

            if self.teach_task.is_done() or self.teach_task.steps>starts_steps+max_steps:
                return {}

            print(f"Searching for {target} near {receptacle_target}...")
            # if args.use_estimated_depth:
            #     self.navigation.bring_head_to_angle(angle=45, vis=self.vis)
            # else:
            #     self.navigation.bring_head_to_angle(update_obs=True, vis=self.vis)
            # centroid = centroids_receptacle[c_idx] # take first one
            # label = labels_receptacle[c_idx]
            object_id = object_ids_receptacle[c_idx]
            # obj_center_camX0 = {'x':centroid[0], 'y':-centroid[1], 'z':centroid[2]}
            # map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0)

            # ind_i, ind_j  = self.navigation.get_interaction_reachable_map_pos(map_pos)

            # self.navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh, search_mode=search_mode) # set point goal in map

            # out = self.navigation.navigate_to_point_goal(
            #     vis=self.vis, 
            #     text=f"Search for {target}", 
            #     object_tracker=self.object_tracker,
            #     search_object=target,
            #     )
            self.navigate(obj_ID=object_id)

            if target is not None:
                centroids, labels, object_ids = self.object_tracker.get_centroids_and_labels(
                    return_ids=True, object_cat=target
                    )

                if len(centroids)>0:
                    print(f"Found {target}!!!")
                    return {'centroids':centroids, 'labels':labels, 'object_ids':object_ids}

            # if receptacle_target in self.OPENABLE_CLASS_LIST:
            #     print("Stepping back...")
            #     self.navigation.step_back(vis=self.vis, text=f"Stepping back", object_tracker=self.object_tracker)
            #     self.navigation.orient_camera_to_point(self.obj_center_camX0, vis=self.vis, text=f"Orient to object", object_tracker=self.object_tracker) 

            # self.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
            # self.navigation.orient_camera_to_point(obj_center_camX0, vis=self.vis, text=f"Orient to {receptacle_target}", object_tracker=self.object_tracker) 

            if receptacle_target in self.OPENABLE_CLASS_LIST:
                print(f"Opening {receptacle_target}...")
                success_open, error = self.execute_action(
                    "Open", 
                    receptacle_target,
                    object_done_holding=False, 
                    retry_image=False, 
                    retry_location=False,
                    step_back_if_open=False,
                    )
                print(f"Open success? {success_open}: {error}")

                if not success_open:
                    success_open, error = self.execute_action(
                    "Open", 
                    receptacle_target,
                    object_done_holding=False, 
                    retry_image=False, 
                    retry_location=False,
                    step_back_if_open=True,
                    )

            print('Searching local region...')
            self.navigation.search_local_region(
                vis=self.vis, 
                text=f"Search for {target}", 
                object_tracker=self.object_tracker,
                search_object=target,
                map_pos=self.map_pos,
                )

            if target is not None:
                centroids, labels, object_ids = self.object_tracker.get_centroids_and_labels(
                    return_ids=True, object_cat=target
                    )

                if len(centroids)>0:
                    print(f"Found {target}!!!")
                    return {'centroids':centroids, 'labels':labels, 'object_ids':object_ids}

            if receptacle_target in self.OPENABLE_CLASS_LIST and success_open:
                # self.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
                # self.navigation.orient_camera_to_point(obj_center_camX0, vis=self.vis, text=f"Orient to {receptacle_target}", object_tracker=None) 
                self.navigate(obj_ID=object_id)
                self.navigation.orient_camera_to_point(self.obj_center_camX0, vis=self.vis, text=f"Orient to object", object_tracker=self.object_tracker) 
                print(f"Closing {receptacle_target}...")
                success, error = self.execute_action(
                    "Close", 
                    receptacle_target,
                    object_done_holding=False, 
                    retry_image=False, 
                    retry_location=False,
                    )
                print(f"Close success? {success}: {error}")
        
        return {}

    def search_near_related_objects(self, object_name, max_steps=150):
        if self.use_llm_search:
            '''
            Get commonsense search locations from LLM
            '''
            self.get_search_objects(object_name)

        starts_steps = self.teach_task.steps

        found_obj = {}
        if object_name in self.search_dict.keys():
            print(f"Searching for {object_name} near the following objects: {self.search_dict[object_name]}")
            search_objects = self.search_dict[object_name]
            for search_obj in search_objects:
                found_obj = self.search_near_object(
                    object_name, 
                    search_obj,
                    max_steps=max_steps - (self.teach_task.steps - starts_steps),
                    steps_cur=starts_steps,
                    )
                if found_obj or self.teach_task.steps>starts_steps+max_steps:
                    break
        
        return found_obj

    def pickup_category(self, object_name):
        
        if self.object_tracker.get_ID_of_holding() is not None:
            # set down any objects the agent is holding
            held_id = self.object_tracker.get_ID_of_holding()
            label_holding = self.object_tracker.objects_track_dict[held_id]["label"]
            print(f"setting down held object {label_holding}...")
            # first try to place object near other object if one in hand
            success_ = self.put_down()
            if success_:
                self.object_tracker.objects_track_dict[held_id]["holding"] = False

        success_nav = self.navigate(object_name)
        subgoal_name = "Pickup"
        object_class = object_name
        retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_class in self.RETRY_DICT_IMAGE[subgoal_name]) else False
        retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_class in self.RETRY_DICT_LOCATION[subgoal_name]) else False
        success_man, error = self.execute_action(
            subgoal_name, 
            object_class, 
            object_done_holding=False, 
            retry_image=retry_image, 
            retry_location=retry_location
            )  
        return subgoal_name and object_class

    def clean(
        self,
        object_id,
        object_class,
    ):  
        '''
        Skill to clean an object
        '''
        
        print("CLEANING!!!")
        self.put_down()
        _, object_center, obj_ID_faucet = self.get_object_data("Faucet")
        if object_center is None:
            return
        if self.object_tracker.objects_track_dict[obj_ID_faucet]["toggled"]:
            self.navigate(obj_ID=obj_ID_faucet)
            subgoal_name = "ToggleOff"
            object_name = "Faucet"
            retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
            retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
            success_open, error = self.execute_action(
                        subgoal_name, 
                        object_name,
                        object_done_holding=False, 
                        retry_image=retry_image, 
                        retry_location=retry_location,
                        )
        print("CLEANING!!!")
        self.navigate(obj_ID=object_id)
        subgoal_name = "Pickup"
        object_name = object_class
        retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
        retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
        success_open, error = self.execute_action(
                    subgoal_name, 
                    object_name,
                    object_done_holding=False, 
                    retry_image=retry_image, 
                    retry_location=retry_location,
                    )
        _, object_center, obj_ID_sink = self.get_object_data("SinkBasin")
        if object_center is None:
            return
        print("CLEANING!!!")
        self.navigate(obj_ID=obj_ID_sink)
        subgoal_name = "Place"
        object_name = "SinkBasin"
        retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
        retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
        success_open, error = self.execute_action(
                    subgoal_name, 
                    object_name,
                    object_done_holding=False, 
                    retry_image=retry_image, 
                    retry_location=retry_location,
                    )
        print("CLEANING!!!")
        self.navigate(obj_ID=obj_ID_faucet)
        subgoal_name = "ToggleOn"
        object_name = "Faucet"
        retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
        retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
        success_open, error = self.execute_action(
                    subgoal_name, 
                    object_name,
                    object_done_holding=False, 
                    retry_image=retry_image, 
                    retry_location=retry_location,
                    )
        print("CLEANING!!!")
        subgoal_name = "ToggleOff"
        object_name = "Faucet"
        retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
        retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
        success_open, error = self.execute_action(
                    subgoal_name, 
                    object_name,
                    object_done_holding=False, 
                    retry_image=retry_image, 
                    retry_location=retry_location,
                    )
        print("CLEANING!!!")
        self.navigate(obj_ID=object_id)
        subgoal_name = "Pickup"
        object_name = object_class
        retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
        retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
        success_open, error = self.execute_action(
                    subgoal_name, 
                    object_name,
                    object_done_holding=False, 
                    retry_image=retry_image, 
                    retry_location=retry_location,
                    )
        print("CLEANING!!!")
        self.put_down()



    def get_object_data(
        self, 
        object_name, 
        grounding_phrase=None, 
        parent_object=None
        ):     
        '''
        object_name: Object category of object to find
        object_phrase: grounding phrase to help locate the object
        parent_object: parent object that the object of interest originated from
        '''

        if parent_object is not None:
            try:
                object_id_parent = int(parent_object.split('_')[-1])
                if "Sliced" in object_name and not object_name.replace("Sliced","")==self.object_tracker.objects_track_dict[object_id_parent]["label"]:
                    assert(False) # move to except side
                attributes = copy.deepcopy(self.object_tracker.objects_track_dict[object_id_parent])
                del attributes["ID"]
                attributes["label"] = object_name
                if args.force_actions and "metaID" in attributes.keys() and "Sliced" in object_name:
                    metaIDs_exist = [self.object_tracker.objects_track_dict[id_]["metaID"] for id_ in self.object_tracker.objects_track_dict.keys() if "metaID" in self.object_tracker.objects_track_dict[id_].keys()]
                    for obj in self.controller.last_event.metadata["objects"]:
                        if attributes["metaID"] in obj["objectId"] and "Sliced" in obj["objectId"] and obj["objectId"] not in metaIDs_exist:
                            attributes["metaID"] = obj["objectId"]
                self.object_tracker.create_new_object_entry(attributes)
                obj_ID = self.object_tracker.id_index-1
                object_center = self.object_tracker.objects_track_dict[obj_ID]["locs"]
                return object_name, object_center, obj_ID
            except:
                pass # carry on with category

        if object_name not in self.include_classes:
            object_name = self.llm.get_closest_category_to_word(object_name, grounding_phrase)

        if "Sliced" in object_name:
            centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(
                return_ids=True, 
                object_cat=object_name.replace('Sliced', ''), 
                include_holding=True,
                check_attrs={'sliced':True},
                )
            if len(IDs)>0:
                object_id_parent = IDs[0]
                # object_center = centroids[0]
                attributes = copy.deepcopy(self.object_tracker.objects_track_dict[object_id_parent])
                del attributes["ID"]
                attributes["label"] = object_name
                self.object_tracker.create_new_object_entry(attributes)
                obj_ID = self.object_tracker.id_index-1
                object_center = self.object_tracker.objects_track_dict[obj_ID]["locs"]
                return object_name, object_center, obj_ID

        centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=object_name, include_holding=True)
        if object_name not in labels:

            if object_name in self.could_not_find:
                print("Already searched for object and could not find.. skipping..")
                return None, None, None

            start_step = self.teach_task.steps
            print(f"Object {object_name} not in memory.. searching for it")
            # 1) search near objects mentioned in the dialogue & commonsense
            found_obj = self.search_near_related_objects(object_name, max_steps=args.max_search_steps)
            # 2) deploy random search
            if not found_obj and not self.use_llm_search:
                found_obj = self.random_search(object_name, max_steps=50)
            end_step = self.teach_task.steps
            print(f"Total searching steps: {end_step-start_step}")
            if not found_obj:
                # still could not find object, so return
                self.could_not_find.append(object_name)
                return None, None, None
            if self.vis is not None:
                for _ in range(5):
                    rgb = np.float32(self.get_image(self.controller).copy())
                    self.vis.add_frame(rgb, text="FOUND OBJECT!!", add_map=self.add_map)
            centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=object_name)

        # first one will be highest confidence
        object_center = centroids[0]
        object_name = labels[0]
        obj_ID = IDs[0]

        return object_name, object_center, obj_ID

    def navigate(
        self, 
        object_name=None, 
        obj_ID=None,
        ):

        if self.teach_task.is_done():
            return False

        success = False

        if object_name is not None:
            object_class, object_center, obj_ID = self.get_object_data(object_name)
        elif obj_ID is not None:
            object_class = self.object_tracker.objects_track_dict[obj_ID]["label"]
            object_center = self.object_tracker.objects_track_dict[obj_ID]["locs"]
        else:
            assert NotImplementedError

        self.navigate_obj_info = {}
        self.navigate_obj_info["object_class"] = object_class
        self.navigate_obj_info["object_center"] = object_center
        self.navigate_obj_info["obj_ID"] = obj_ID

        if object_center is None:
            if self.object_tracker.get_label_of_holding()==object_name:
                print("Holding desired object! Can't navigate to it!")
                return True
            if self.vis is not None:
                for _ in range(5):
                    rgb = np.float32(self.get_image(self.controller).copy())
                    self.vis.add_frame(rgb, text="COULD NOT FIND OBJECT")
            # continue
            return success

        success = True
        
        print("Note: subtracting object y by agent height before point nav")
        obj_center_camX0_ = {'x':object_center[0], 'y':-object_center[1], 'z':object_center[2]}
        self.obj_center_camX0 = obj_center_camX0_
        map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
        self.map_pos = map_pos

        ind_i, ind_j  = self.navigation.get_clostest_reachable_map_pos(map_pos)

        self.navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh)
        self.navigation.navigate_to_point_goal(vis=self.vis, text=f"Navigate to {object_class}", object_tracker=self.object_tracker, max_fail=5, add_obs=True)

        self.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
        self.navigation.orient_camera_to_point(obj_center_camX0_, vis=self.vis, text=f"Orient to {object_class}", object_tracker=self.object_tracker)  

        return success

    def put_down(self, topk_objects=2, topk_tries=3, object_done_holding=False):
        if self.object_tracker.get_ID_of_holding() is None:
            return True

        centroids_target, labels = self.object_tracker.get_centroids_and_labels(return_ids=False)
        general_rec_in_scene = [l for l in self.general_receptacles_classes if l in labels]
        general_rec_in_scene = general_rec_in_scene[:topk_objects] # try up to topk

        success = False
        subgoal_name = "Place"
        # place object in any available receptacle
        while not success and len(general_rec_in_scene)>0:
            object_name = general_rec_in_scene.pop(0)
            centroids, labels, IDs = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=object_name, include_holding=False)
            unique_obj_tries = min(len(centroids), topk_tries)
            for try_i in range(unique_obj_tries):
                self.navigate(object_name)
                retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
                retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
                success, error = self.execute_action(
                        subgoal_name, 
                        object_name, 
                        object_done_holding=object_done_holding, 
                        retry_image=retry_image, 
                        retry_location=retry_location
                        ) 
                if success:
                    # self.object_tracker.objects_track_dict[obj_ID]["holding"] = True
                    break

        return success

    def empty(
        self,
        object_name=None,
        obj_ID=None,
        # max_fails=5,
    ):
        '''
        empty out the object / clear the object from other objects
        '''
        # start_fails = self.teach_task.num_fails

        if self.teach_task.num_fails>=int((2/3)*self.teach_task.max_fails):
            return True

        if args.simulate_actions:
            return True

        if obj_ID is not None and not type(obj_ID)==int:
            obj_ID = int(obj_ID)

        if object_name is None:
            object_name = self.object_tracker.get_label_from_ID(obj_ID)

        print(f"Begin emptying out {object_name}...")

        if self.object_tracker.get_ID_of_holding() is not None:
            # set down any objects the agent is holding
            held_id = self.object_tracker.get_ID_of_holding()
            label_holding = self.object_tracker.objects_track_dict[held_id]["label"]
            print(f"setting down held object {label_holding}...")
            # first try to place object near other object if one in hand
            success_ = self.put_down()
            if success_:
                self.object_tracker.objects_track_dict[held_id]["holding"] = False
        else:
            held_id = None

        if object_name in ["Sink", "SinkBasin"]:
            # turn off faucet if on
            centroids_target_faucet, labels_target_faucet, object_ids_faucet = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat="Faucet")
            if object_ids_faucet:
                # pickup
                obj_ID_faucet = int(object_ids_faucet[0])
                if self.object_tracker.objects_track_dict[obj_ID_faucet]["toggled"]:
                    self.navigate(obj_ID=obj_ID_faucet)
                    subgoal_name = "ToggleOff"
                    object_name_faucet = "Faucet"
                    retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name_faucet in self.RETRY_DICT_IMAGE[subgoal_name]) else False
                    retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name_faucet in self.RETRY_DICT_LOCATION[subgoal_name]) else False
                    success_toggle_off, error = self.execute_action(
                                subgoal_name, 
                                object_name_faucet,
                                object_done_holding=False, 
                                retry_image=retry_image, 
                                retry_location=retry_location,
                                )
                    if success_toggle_off:
                        self.object_tracker.objects_track_dict[obj_ID_faucet]["toggled"] = False

        if self.teach_task.num_fails>=int((2/3)*self.teach_task.max_fails):
            return True

        # elif object_name in ["Toaster"]:
        #     # if its already toggled then return
        #     if self.object_tracker.objects_track_dict[obj_ID]["toggled"]:
        #         return True

        if object_name in self.OPENABLE_OBJECTS:
            if object_name in self.TOGGLEABLE and self.object_tracker.objects_track_dict[obj_ID]["toggled"]:
                self.navigate(obj_ID=obj_ID)
                subgoal_name = "ToggleOff"
                retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
                retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
                success_toggle, error = self.execute_action(
                            subgoal_name, 
                            object_name,
                            object_done_holding=False, 
                            retry_image=retry_image, 
                            retry_location=retry_location,
                            )
                if success_toggle:
                    self.object_tracker.objects_track_dict[obj_ID]["toggled"] = False
            # open microwave
            if not self.object_tracker.objects_track_dict[obj_ID]["open"]:
                self.navigate(obj_ID=obj_ID)
                subgoal_name = "Open"
                retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name in self.RETRY_DICT_IMAGE[subgoal_name]) else False
                retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name in self.RETRY_DICT_LOCATION[subgoal_name]) else False
                success_open, error = self.execute_action(
                            subgoal_name, 
                            object_name,
                            object_done_holding=False, 
                            retry_image=retry_image, 
                            retry_location=retry_location,
                            )
                if success_open:
                    self.object_tracker.objects_track_dict[obj_ID]["open"] = True

        if self.teach_task.num_fails>=int((2/3)*self.teach_task.max_fails):
            return True

        if obj_ID is not None:
            # first detect any objects near the target
            self.search_near_object(None, obj_ID)
            centroids_target = np.asarray([self.object_tracker.objects_track_dict[obj_ID]['locs']])
        else:
            # first detect any objects near the target
            self.search_near_object(None, object_name)
            # next get all objects on/in the object by distance thresholding
            centroids_target, _, _ = self.object_tracker.get_centroids_and_labels(return_ids=True, object_cat=object_name)
        IDs_threshold = self.object_tracker.get_IDs_within_distance(centroids_target, 0.5)
        labels_threshold = [self.object_tracker.get_label_from_ID(id_) for id_ in IDs_threshold]
        # print(0, labels_threshold)
        if obj_ID is not None:
            if 'supporting' in self.object_tracker.objects_track_dict[obj_ID].keys() and self.object_tracker.objects_track_dict[obj_ID]["supporting"] is not None:
                '''
                Only keep objects that we know the target is supporting
                '''
                list_B = set(self.object_tracker.objects_track_dict[obj_ID]["supporting"])
                list_A = labels_threshold
                both = []
                for i in range(len(list_A)):
                    if list_A[i] in list_B:
                        both.append(i)
                IDs_threshold = [IDs_threshold[idx] for idx in both]
                labels_threshold = [labels_threshold[idx] for idx in both]
        # print(1, labels_threshold)
        valid_objects = [True if l in self.PICKUPABLE_OBJECTS else False for l in labels_threshold]
        IDs_threshold = [IDs_threshold[idx] for idx in range(len(valid_objects)) if valid_objects[idx]]
        labels_threshold = [labels_threshold[idx] for idx in range(len(valid_objects)) if valid_objects[idx]]
        for obj_id_idx in range(len(IDs_threshold)):

            if self.teach_task.num_fails>=int((2/3)*self.teach_task.max_fails):
                return True

            # pickup
            obj_ID_target = IDs_threshold[obj_id_idx]
            self.navigate(obj_ID=obj_ID_target)
            subgoal_name = "Pickup"
            object_name_ = labels_threshold[obj_id_idx]
            print(f"Moving {object_name_} out of {object_name}...")
            retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name_ in self.RETRY_DICT_IMAGE[subgoal_name]) else False
            retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name_ in self.RETRY_DICT_LOCATION[subgoal_name]) else False
            success, error = self.execute_action(
                    subgoal_name, 
                    object_name_, 
                    object_done_holding=False, 
                    retry_image=retry_image, 
                    retry_location=retry_location
                    )  
            
            # move away from object_name
            if success:
                self.object_tracker.objects_track_dict[obj_ID_target]["holding"] = True
                success_ = self.put_down()
                if success_:
                    self.object_tracker.objects_track_dict[obj_ID_target]["holding"] = False
            
        if held_id is not None:
            self.navigate(obj_ID=held_id)
            subgoal_name = "Pickup"
            object_name_ = self.object_tracker.get_label_from_ID(held_id)
            print(f"picking up previously held object {object_name_}...")
            retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name_ in self.RETRY_DICT_IMAGE[subgoal_name]) else False
            retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name_ in self.RETRY_DICT_LOCATION[subgoal_name]) else False
            success, error = self.execute_action(
                    subgoal_name, 
                    object_name_, 
                    object_done_holding=False, 
                    retry_image=retry_image, 
                    retry_location=retry_location
                    )  
            if success:
                self.object_tracker.objects_track_dict[held_id]["holding"] = True

        return True # pretend it succeeded even if it didn't - error correction should take care of this if failed

    # def clean(
    #     self,
    #     obj_ID,
    # ):  
        
    #     object_name = self.object_tracker.get_label_from_ID(obj_ID)
    #     print(f"Starting to clean {object_name}")
    #     if self.object_tracker.get_ID_of_holding() is None:
    #         print(f"Picking up {}")
    #         self.navigate(obj_ID=obj_ID)
    #         subgoal_name = "Pickup"
    #         object_name_ = self.object_tracker.get_label_from_ID(held_id)
    #         print(f"picking up previously held object {object_name_}...")
    #         retry_image = True if (subgoal_name in self.RETRY_ACTIONS_IMAGE and object_name_ in self.RETRY_DICT_IMAGE[subgoal_name]) else False
    #         retry_location = True if (subgoal_name in self.RETRY_ACTIONS_LOCATION and object_name_ in self.RETRY_DICT_LOCATION[subgoal_name]) else False
    #         success, error = self.execute_action(
    #                 subgoal_name, 
    #                 object_name_, 
    #                 object_done_holding=False, 
    #                 retry_image=retry_image, 
    #                 retry_location=retry_location
    #                 )  
    #         if success:
    #             self.object_tracker.objects_track_dict[held_id]["holding"] = True

    def run_corrective_action(
        self,
        action_name: str,
    ): 
        '''
        Deploy corrective action protocol
        '''
        print(f"Taking corrective action: {action_name}")
        if action_name=="MoveBack":
            self.navigation.step_back(vis=self.vis, text=f"Stepping back", object_tracker=self.object_tracker)
        elif action_name=="MoveCloser":
            if self.obj_center_camX0 is not None:
                map_pos = self.navigation.get_map_pos_from_aithor_pos(self.obj_center_camX0)
                ind_i, ind_j  = self.navigation.get_interaction_reachable_map_pos(map_pos, location_quandrant='third')
                self.navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh)
                self.navigation.navigate_to_point_goal(vis=self.vis, text=f"Navigate closer", object_tracker=self.object_tracker, max_fail=5, add_obs=True)
                self.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
                self.navigation.orient_camera_to_point(self.obj_center_camX0, vis=self.vis, text=f"Orient to object", object_tracker=self.object_tracker) 
            else:
                self.navigation.take_action("MoveAhead")
        elif action_name=="MoveAlternate":
            if self.obj_center_camX0 is not None:
                map_pos = self.navigation.get_map_pos_from_aithor_pos(self.obj_center_camX0)
                ind_i, ind_j  = self.navigation.get_interaction_reachable_map_pos(map_pos, location_quandrant='third')
                self.navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh)
                self.navigation.navigate_to_point_goal(vis=self.vis, text=f"Navigate to alternate", object_tracker=self.object_tracker, max_fail=5, add_obs=True)
                self.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
                self.navigation.orient_camera_to_point(self.obj_center_camX0, vis=self.vis, text=f"Orient to object", object_tracker=self.object_tracker)
            else:
                self.navigation.take_action("RotateRight")
                self.navigation.take_action("MoveAhead")
                self.navigation.take_action("RotateLeft")
        else:
            assert(False) # what subgoal is this?

        return True

    def execute_action(
        self, 
        action_name: str, 
        object_name: str, 
        object_done_holding=False,
        retry_image=True, # Retry interaction at multiple keypoints in the image
        retry_location=True, # Retry interaction at multiple locations around the object
        log_error=True, # log error messages in teach base
        remove_obj_fail=True,
        step_back_if_open=True,
        ):

        if self.teach_task.is_done():
            return False, "task done"

        if action_name=="PutDown":
            step_success = self.put_down(object_done_holding=object_done_holding)
            return step_success, "No error message here, using Teach Wrapper"

        text_ = ''
        print("action name: ", action_name)
        print("Target object: ", object_name)

        object_class = self.navigate_obj_info["object_class"]
        object_center = self.navigate_obj_info["object_center"]
        obj_ID = self.navigate_obj_info["obj_ID"]
        if object_center is None:
            return False, "Object not found in memory, need more exploration."

        if action_name=="Open" and step_back_if_open and not args.simulate_actions:
            self.navigation.step_back(vis=self.vis, text=f"Stepping back", object_tracker=self.object_tracker) 
            self.navigation.orient_camera_to_point(self.obj_center_camX0, vis=self.vis, text=f"Orient to object", object_tracker=self.object_tracker) 

        if args.force_actions and "metaID" in self.object_tracker.objects_track_dict[obj_ID].keys() and not args.simulate_actions:
            '''
            Open parent object if desired object is contained.
            '''
            obj_dict = {}
            for obj in self.controller.last_event.metadata["objects"]:
                obj_dict.update({obj["objectId"]:obj})
            obj_dict_meta = {}
            for k in self.object_tracker.objects_track_dict.keys():
                obj_dict_meta[self.object_tracker.objects_track_dict[k]["metaID"]] = k
            objectMetaId = self.object_tracker.objects_track_dict[obj_ID]["metaID"]
            obj_meta = obj_dict[objectMetaId]
            parent_receptacles = obj_meta['parentReceptacles']
            p_recs = []
            if parent_receptacles is not None:
                for p_rec in parent_receptacles:
                    rec_type = p_rec.split('|')[0]
                    if rec_type in self.OPENABLE_OBJECTS and not obj_dict[p_rec]['isOpen']:
                        self.navigation.step_back(vis=self.vis, text=f"Stepping back", object_tracker=self.object_tracker)
                        self.navigation.orient_camera_to_point(self.obj_center_camX0, vis=self.vis, text=f"Orient to object", object_tracker=self.object_tracker)
                        print(f"Opening {p_rec} before {action_name} {object_name}...")
                        # open receptacle
                        event = self.controller.step(
                                action="OpenObject",
                                objectId=p_rec,
                                forceAction=True
                            )
                        if event.metadata["lastActionSuccess"] and p_rec in obj_dict_meta.keys():
                            self.object_tracker.objects_track_dict[obj_dict_meta[p_rec]]["open"] = True
                            p_recs.append(p_rec)
                            obj_center_camX0_prec = self.obj_center_camX0
                        # plt.figure()
                        # plt.imshow(self.controller.last_event.frame)
                        # plt.savefig('output/test.png')
                        # for _ in range(5):
                        #     rgb = np.float32(self.get_image(self.controller))
                        #     self.vis.add_frame(rgb, text=f"OPEN", add_map=self.add_map)
                        # self.vis.render_movie(os.path.join(self.output_folder, self.folder_tag), 0, tag=f"OPENING_movie_environment{self.environment_index}_refinement{self.refinement_attempt}_{self.tag}")
                        # st()

        def get_point_2Ds(sampling='center'):
            if self.use_GT_seg_for_interaction:
                point_2Ds, msg = self.object_tracker.get_2D_point(
                    object_category=object_name, 
                    object_id=obj_ID,
                    sampling=sampling
                )
                method_text = "GT SEGMENTATION"
            else:
                camX0_T_camX = self.navigation.explorer.get_camX0_T_camX()
                obj_center_camX0_ = {'x':object_center[0], 'y':object_center[1], 'z':object_center[2]}
                camX_T_camX0 = utils.geom.safe_inverse_single(camX0_T_camX)
                point_2Ds, method_text = self.object_tracker.get_2D_point(
                    camX_T_origin=camX_T_camX0, 
                    obj_center_camX0_=obj_center_camX0_, 
                    object_category=object_name, 
                    rgb=self.get_image(self.controller), 
                    score_threshold=0.0,
                    object_id=obj_ID,
                    sampling=sampling
                    )
            return point_2Ds, method_text
        point_2Ds, method_text = get_point_2Ds()

        if point_2Ds is None:
            point_2Ds = [] # none detected
        elif object_name in ["StoveKnob"]:
            pass
        else:
            point_2Ds = point_2Ds[0:1] 

        if args.simulate_actions:
            point_2Ds = [None]

        step_success = False
        num_tries = 1
        err_message = ''
        help_message = ''
        for point_2D_idx in range(len(point_2Ds)):
            point_2D = point_2Ds[point_2D_idx]

            if self.vis is not None and self.render and not args.simulate_actions:
                for _ in range(5):
                    rgb = np.float32(self.get_image(self.controller))
                    rgb = cv2.circle(rgb, (int(point_2D[1]),int(point_2D[0])), radius=5, color=(0, 0, 255),thickness=2)
                    self.vis.add_frame(rgb, text=f"{object_name} 2D point identified with:"+method_text, add_map=self.add_map)
            
            if not args.simulate_actions:
                point_2D_ = point_2D.copy()
                point_2D_[0] = point_2D_[0] / self.web_window_size
                point_2D_[1] = point_2D_[1] / self.web_window_size        
                obj_relative_coord = [max(min(point_2D_[0], 0.998), 0.001), max(min(point_2D_[1], 0.998), 0.001)]

                # agent_pos = np.asarray(list(self.navigation.explorer.position.values()))
                agent_pos = self.navigation.explorer.get_camX0_T_camX()[:3,3].numpy()
                distance_to_object = np.sqrt(np.sum((object_center[[0,2]] - agent_pos[[0,2]])**2)) # only use x and z for distance
                # distance_to_object = np.sqrt(np.sum((object_center - agent_pos)**2))
                print(f"Distance to object: {distance_to_object}")
            if args.simulate_actions:
                step_success = True
            elif distance_to_object<self.visibility_distance+0.5:

                # attempt interaction
                prev_image = self.get_image(self.controller).copy()
                sim_succ = self.teach_task.step(action_name, obj_relative_coord, object_name, log_error=log_error)

                rgb = self.get_image(self.controller)
                step_success = self.teach_task.action_success()

                err_message += f'{self.teach_task.err_message} '
                help_message += f'{self.teach_task.help_message} '
                num_tries += 1

                if self.vis is not None and self.render:
                    if not step_success:
                        text=f"{object_name} {action_name} Percieved Fail. (Actual Success? {sim_succ})"
                    else:
                        text=f"Percieved action {object_name} {action_name} SUCCESS!!"
                    for _ in range(5):
                        rgb = np.float32(self.get_image(self.controller))
                        rgb = cv2.circle(rgb, (int(point_2D_[1]* self.web_window_size),int(point_2D_[0]* self.web_window_size)), radius=5, color=(0, 0, 255),thickness=2)
                        self.vis.add_frame(rgb, text=text+text_, add_map=self.add_map)
                
                if not step_success and retry_image:
                    '''
                    Randomly try other locations in the mask
                    '''
                    num_offsets = 4
                    offsets = [[0, 0], [0, 0], [0, 0], [0, 0]]
                    # offsets = [[0, 40], [40, 0], [0, -40], [-40, 0]]
                    
                    num_tries_place = 1

                    for mult in range(1, num_tries_place+1):
                        for offset in offsets:
                            point_2Ds_, method_text = get_point_2Ds(sampling='random')
                            point_2D = point_2Ds_[point_2D_idx]
                            point_2D_ = point_2D.copy()
                            point_2D_ = [point_2D_[0] + offset[0]*mult, point_2D_[1] + offset[1]*mult]
                            point_2D_[0] = point_2D_[0] / self.web_window_size
                            point_2D_[1] = point_2D_[1] / self.web_window_size        
                            obj_relative_coord_ = [max(min(point_2D_[0], 0.998), 0.001), max(min(point_2D_[1], 0.998), 0.001)]
                            sim_succ = self.teach_task.step(action_name, obj_relative_coord_, object_name, log_error=True)
                            rgb = self.get_image(self.controller)
                            step_success = self.teach_task.action_success()
                            if self.vis is not None and self.render:
                                if not step_success:
                                    text=f"{object_name} {action_name} Percieved Fail. (Actual Success? {sim_succ})"
                                else:
                                    text=f"Percieved action {object_name} {action_name} SUCCESS!!"
                                for _ in range(5):
                                    rgb = np.float32(self.get_image(self.controller))
                                    rgb = cv2.circle(rgb, (int(point_2D_[1]* self.web_window_size),int(point_2D_[0]* self.web_window_size)), radius=5, color=(0, 0, 255),thickness=2)
                                    self.vis.add_frame(rgb, text=text+text_, add_map=self.add_map)
                            if step_success:
                                break
                        if step_success:
                            break
            else:
                # object too far from agent to interact
                step_success = False
                self.teach_task.err_message = "Object is too far from agent."
                self.teach_task.help_message = "Object must be within 1.5 meters to interact."
                if self.vis is not None and self.render:
                    text=f"Object too far!"
                    for _ in range(5):
                        rgb = np.float32(self.get_image(self.controller))
                        rgb = cv2.circle(rgb, (int(point_2D_[1]* self.web_window_size),int(point_2D_[0]* self.web_window_size)), radius=5, color=(0, 0, 255),thickness=2)
                        self.vis.add_frame(rgb, text=text+text_, add_map=self.add_map)

            step_success_outside = False
            if not step_success and retry_location and self.obj_center_camX0 is not None: # and self.teach_task.num_fails<int((2/3)*self.teach_task.max_fails):
                print("Trying from a second location...")
                map_pos = self.navigation.get_map_pos_from_aithor_pos(self.obj_center_camX0)
                ind_i, ind_j  = self.navigation.get_interaction_reachable_map_pos(map_pos, location_quandrant='second')
                self.navigation.set_point_goal(ind_i, ind_j, dist_thresh=0.3)
                steps_before = self.teach_task.steps
                self.navigation.navigate_to_point_goal(vis=self.vis, text=f"Navigate to {object_name}", object_tracker=self.object_tracker, max_fail=5, add_obs=True)
                self.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
                self.navigation.orient_camera_to_point(self.obj_center_camX0, vis=self.vis, text=f"Orient to {object_name}", object_tracker=self.object_tracker) 
                steps_after = self.teach_task.steps
                if steps_after - steps_before > 0:
                    step_success, _ = self.execute_action(
                            action_name=action_name, 
                            object_name=object_name, 
                            object_done_holding=object_done_holding,
                            retry_image=retry_image,
                            retry_location=False,
                            log_error=True,
                            remove_obj_fail=False,
                            ) 
                    step_success_outside = step_success
                    err_message += f'{self.teach_task.err_message} '
                    help_message += f'{self.teach_task.help_message} '
                    num_tries += 1
                if not step_success:
                    print("Trying from a third location...")
                    map_pos = self.navigation.get_map_pos_from_aithor_pos(self.obj_center_camX0)
                    ind_i, ind_j  = self.navigation.get_interaction_reachable_map_pos(map_pos, location_quandrant='third')
                    self.navigation.set_point_goal(ind_i, ind_j, dist_thresh=0.3)
                    steps_before = self.teach_task.steps
                    self.navigation.navigate_to_point_goal(vis=self.vis, text=f"Navigate to {object_name}", object_tracker=self.object_tracker, max_fail=5, add_obs=True)
                    self.navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
                    self.navigation.orient_camera_to_point(self.obj_center_camX0, vis=self.vis, text=f"Orient to {object_name}", object_tracker=self.object_tracker) 
                    steps_after = self.teach_task.steps
                    if steps_after - steps_before > 0:
                        step_success, _ = self.execute_action(
                                action_name=action_name, 
                                object_name=object_name, 
                                object_done_holding=object_done_holding,
                                retry_image=retry_image,
                                retry_location=False,
                                log_error=True,
                                remove_obj_fail=False,
                                ) 
                        step_success_outside = step_success
                        err_message += f'{self.teach_task.err_message} '
                        help_message += f'{self.teach_task.help_message} '
                        num_tries += 1

            # Update tracker
            if (step_success and not step_success_outside) and action_name=="Place":
                #centroid is at the point of placement

                if not args.simulate_actions:
                    # get 3D placement location
                    pad = 2
                    camX0_T_camX = self.navigation.explorer.get_camX0_T_camX()
                    _, depth = self.navigation.get_obs(adjust_depth=False) #self.navigation.get_obs(head_tilt=self.navigation.explorer.head_tilt)
                    # if depth.shape[0]!=self.W:
                    #     depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
                    depth = np.nan_to_num(depth)
                    # points2d1 = np.clip(np.arange(int(point_2D_[1]*self.web_window_size)-pad, int(point_2D_[1]*self.web_window_size)+pad), a_min=0, a_max=self.W-1)
                    # points2d2 = np.clip(np.arange(int(point_2D_[0]*self.web_window_size)-pad, int(point_2D_[0]*self.web_window_size)+pad), a_min=0, a_max=self.H-1)
                    # arg_idx = np.argsort(depth[points2d1,points2d2])[len(depth[points2d1,points2d2])//2]
                    points2d_idx1 = int(np.clip(point_2D_[1]*self.web_window_size, a_min=0, a_max=self.W-1)) #points2d1[arg_idx]
                    points2d_idx2 = int(np.clip(point_2D_[0]*self.web_window_size, a_min=0, a_max=self.W-1)) #points2d2[arg_idx]
                    depth_ = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).cuda()
                    xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(self.pix_T_camX).unsqueeze(0).float().cuda())
                    xyz_origin = utils.geom.apply_4x4(camX0_T_camX.float().cuda(), xyz.float()).squeeze().cpu().numpy()
                    xyz_origin = xyz_origin.reshape(1,self.W,self.H,3) 
                    c_depth = np.squeeze(xyz_origin[:,points2d_idx2, points2d_idx1,:])
                    # st()

                    # plt.figure()
                    # center2D = self.object_tracker.get_reprojected_point_given_ID(obj_ID)
                    # point_2D_ = point_2D.copy()
                    # point_2D_[0] = point_2D_[0] / self.web_window_size
                    # point_2D_[1] = point_2D_[1] / self.web_window_size 
                    # rgb = np.float32(self.get_image(self.controller))
                    # rgb = cv2.circle(rgb, (int(point_2D_[1]* self.web_window_size),int(point_2D_[0]* self.web_window_size)), radius=5, color=(0, 0, 255),thickness=2)  
                    # plt.imshow(rgb.astype(np.uint8))
                    # plt.savefig('output/test.png')

                holding_ID = self.object_tracker.get_ID_of_holding()
                if holding_ID is not None:
                    # for k in self.object_tracker.objects_track_dict.keys():
                    #     if self.object_tracker.objects_track_dict[holding_ID]["holding"]:
                    #         print(self.object_tracker.objects_track_dict[holding_ID])
                    # st()
                    if not args.simulate_actions:
                        self.object_tracker.objects_track_dict[holding_ID]["locs"] = c_depth
                    self.object_tracker.objects_track_dict[holding_ID]["holding"] = False
                    self.object_tracker.objects_track_dict[holding_ID]["scores"] = 1.01
                    if object_done_holding:
                        print("Removing placed item from interaction consideration..")
                        self.object_tracker.objects_track_dict[holding_ID]["can_use"] = False
                        self.object_tracker_ids_removed.append(holding_ID)
            elif (step_success and not step_success_outside) and action_name=="Open":
                self.object_tracker.objects_track_dict[obj_ID]["open"] = True
            elif (step_success and not step_success_outside) and action_name=="Close":
                self.object_tracker.objects_track_dict[obj_ID]["open"] = False
            elif (step_success and not step_success_outside) and action_name=="ToggleOn":
                self.object_tracker.objects_track_dict[obj_ID]["toggled"] = True
            elif (step_success and not step_success_outside) and action_name=="ToggleOff":
                self.object_tracker.objects_track_dict[obj_ID]["toggled"] = False
            elif (step_success and not step_success_outside) and action_name=="Pickup":
                
                self.object_tracker.objects_track_dict[obj_ID]["locs"] = None
                self.object_tracker.objects_track_dict[obj_ID]["holding"] = True
                self.object_tracker.objects_track_dict[obj_ID]["scores"] = 1.01
                if "metaID" in self.object_tracker.objects_track_dict[obj_ID].keys():
                    if "Sliced" in self.object_tracker.objects_track_dict[obj_ID]["label"]:
                        objs = self.controller.last_event.metadata["objects"]
                        for obj in objs:
                            if obj['pickupable'] and obj['isPickedUp']:
                                self.object_tracker.objects_track_dict[obj_ID]["metaID"] = obj['objectId']
            elif (step_success and not step_success_outside) and action_name=="Slice":
                # attributes = {
                #         'locs':self.object_tracker.objects_track_dict[obj_ID]["locs"],
                #         'label':self.object_tracker.objects_track_dict[obj_ID]["label"]+"Sliced",
                #         'scores':1.01,
                #         'holding':False,
                #         'can_use':True,
                #         'sliced':True,
                #         # 'toasted':False,
                #         'dirty':self.object_tracker.objects_track_dict[obj_ID]["dirty"],
                #         'cooked':self.object_tracker.objects_track_dict[obj_ID]["cooked"],
                #         }
                # if "metaID" in self.object_tracker.objects_track_dict[obj_ID].keys():
                #     attributes["metaID"] = self.object_tracker.objects_track_dict[obj_ID]["metaID"]
                # for slice_idx in range(6): # slicing creates 6 new sliced objects
                #     self.object_tracker.create_new_object_entry(attributes)
                # self.object_tracker.objects_track_dict[obj_ID]["can_use"] = False # object has now been sliced and so its not usable directly
                self.object_tracker.objects_track_dict[obj_ID]["sliced"] = True
                # pass

            if args.force_actions and "metaID" in self.object_tracker.objects_track_dict[obj_ID].keys() and not args.simulate_actions:
                '''
                Close parent object if desired object was contained.
                '''
                if p_recs:
                    for p_rec in p_recs:
                        rec_type = p_rec.split('|')[0]
                        # self.navigation.step_back(vis=self.vis, text=f"Stepping back", object_tracker=self.object_tracker)
                        self.navigation.orient_camera_to_point(obj_center_camX0_prec, vis=self.vis, text=f"Orient to object", object_tracker=self.object_tracker)
                        print(f"Closing {p_rec} after {action_name} {object_name}...")
                        # open receptacle
                        event = self.controller.step(
                                action="CloseObject",
                                objectId=p_rec,
                                forceAction=True
                            )
                        if event.metadata["lastActionSuccess"] and p_rec in obj_dict_meta.keys():
                            self.object_tracker.objects_track_dict[obj_dict_meta[p_rec]]["open"] = False
            
            if not step_success and obj_ID is not None and remove_obj_fail: # and (object_name not in self.NONREMOVABLE_CLASSES) and (retry_image or retry_location) and (not self.object_tracker.objects_track_dict[obj_ID]["sliced"]):
                scores = self.object_tracker.get_score_of_label(object_name)
                min_score = min(scores)
                self.object_tracker.objects_track_dict[obj_ID]["scores"] = min_score - 0.01 # move to bottom of totem pole

            self.err_message = err_message
            self.help_message = help_message

        # if not step_success and action_name=="Place" and object_name=="Bowl":
        #     rgb = np.float32(self.get_image(self.controller))
        #     rgb = cv2.circle(rgb, (int(point_2D[1]),int(point_2D[0])), radius=5, color=(0, 0, 255),thickness=2)
        #     plt.figure()
        #     plt.imshow(rgb.astype(np.uint8))
        #     plt.savefig('output/test.png')
        #     st()
        #     for obj in self.controller.last_event.metadata["objects"]:
        #         if obj['objectType']=="Bowl":
        #             print(obj)

        #     self.controller.step(
        #         action="LookDown",
        #         # objectId='Bowl|-01.86|+00.92|+01.80',
        #         forceAction=False,
        #     )

        #     self.controller.step(
        #         action="PutObject",
        #         objectId='Bowl|-01.86|+00.92|+01.80',
        #         forceAction=True,
        #     )

        self.step_success = step_success
        print(f"Action {action_name} success? {step_success}")
        return step_success, "No error message here, using Teach Wrapper"

    