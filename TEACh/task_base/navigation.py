from time import sleep
from arguments import args
import numpy as np
from map_and_plan.mess.utils import Foo
from map_and_plan.mess.explore import Explore
from argparse import Namespace
import logging
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import utils.aithor
import torch
import ipdb
st = ipdb.set_trace
import sys
from PIL import Image, ImageDraw
import os
import cv2
from utils.aithor import compute_metrics, read_task_data
# from task_base.depth import Depth
from task_base.depth import Depth_ZOE as Depth

class Navigation():

    def __init__(
        self, 
        z=None, 
        keep_head_down=False, 
        keep_head_straight=False, 
        look_down_init=True, 
        block_map_positions_if_fail=False,
        controller=None, 
        estimate_depth=True, 
        add_obs_when_navigating_if_explore_fail=False, 
        on_aws=False, 
        max_steps=500, 
        search_pitch_explore=False,
        pix_T_camX=None,
        task=None,
        depth_estimation_network=None,
        ):  

        self.on_aws = on_aws
        print("Initializing NAVIGATION...")
        print("NOTICE: Make sure snapToGrid=False in the Ai2Thor Controller.")  
        self.action_fail_count = 0
        self.controller = controller

        self.W, self.H = args.W, args.H

        self.max_steps = max_steps
        self.obs = Namespace()

        if z is None:
            if args.use_estimated_depth:
                if (0):
                    self.z = [0.4] + list(np.round(np.arange(0.45, 2.06+0.05, 0.05),2))
                    self.obs.camera_aspect_ratio = [args.frame_height, args.frame_width] # <- USE THIS WITH FILM DEPTH ESTIMATION
                else:
                    self.z = [0.8] + list(np.round(np.arange(0.85, 2.06+0.05, 0.05),2))
                    self.obs.camera_aspect_ratio = [args.H, args.W]
            else:
                self.z = [0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
                self.obs.camera_aspect_ratio = [args.H, args.W]

        else:
            self.z = z
        
        
        self.pix_T_camX = pix_T_camX
        self.obs.STEP_SIZE = args.STEP_SIZE 
        self.obs.DT = args.DT 
        self.obs.HORIZON_DT = args.HORIZON_DT 
        self.obs.camera_height = 1.5759992599487305 # fixed when standing up 
        
        self.obs.camera_field_of_view = args.fov
        self.obs.head_tilt_init = 0 
        self.obs.reward = 0
        self.obs.goal = Namespace(metadata={"category": 'cover'})
        actions = ['MoveAhead', 'MoveLeft', 'MoveRight', 'MoveBack', 'RotateRight', 'RotateLeft']
        self.actions = {i:actions[i] for i in range(len(actions))}
        self.action_mapping = {"Pass":0, "MoveAhead":1, "MoveLeft":2, "MoveRight":3, "MoveBack":4, "RotateRight":5, "RotateLeft":6, "LookUp":9, "LookDown":10}
        self.keep_head_down = keep_head_down
        self.keep_head_straight = keep_head_straight
        self.search_pitch_explore = search_pitch_explore
        self.block_map_positions_if_fail = block_map_positions_if_fail
        self.add_obs_when_navigating_if_explore_fail = add_obs_when_navigating_if_explore_fail
        self.add_obs_when_navigating = False
        self.look_down_init = look_down_init
        self.explorer = None
        self.task = task
        self.init_called = False
        self.depth_estimator = None
        if args.use_estimated_depth or args.use_mask_rcnn_pred:
            if depth_estimation_network is None:
                self.depth_estimator = Depth(task=self.task)
            else:
                self.depth_estimator = depth_estimation_network
                self.depth_estimator.task = self.task
                self.depth_estimator.agent = self.task

        self.mapping_images = []

    def init_navigation(self,bounds):
        if not self.init_called:
            self.init_called = True
            self.step_count = 0
            self.goal = Foo()
            self.parse_goal(self.obs.goal)
            logging.error(self.goal.description)
            
            self.rng = np.random.RandomState(0)

            self.cover = False
            if self.goal.category == 'cover' or self.goal.category == 'point_nav':
                print("Z is:", self.z)
                self.explorer = Explore(
                    self.obs, 
                    self.goal, 
                    bounds, 
                    z=self.z, 
                    keep_head_down=self.keep_head_down, 
                    keep_head_straight=self.keep_head_straight, 
                    look_down_init=self.look_down_init,
                    search_pitch_explore=self.search_pitch_explore,
                    block_map_positions_if_fail=self.block_map_positions_if_fail,
                    )
                self.cover = True 
        else:
            print("NAVIGATION INIT ALREADY CALLED!")
            print("Skipping call of navigation since it was already called...")

    def parse_goal(self, goal, fig=None):
        self.goal.category = goal.metadata['category']
        self.goal.description = self.goal.category
        if self.goal.category == 'point_nav':
            self.goal.targets = goal.metadata['targets']
        else:
            self.goal.targets = ['dummy']
 
    def act(self, add_obs=True, action=None, fig=None, point_goal=None):
        self.step_count += 1
        if self.step_count == 1:
            pass

        if self.cover:
            action, param = self.explorer.act(self.obs, action=action, fig=fig, point_goal=point_goal, add_obs=add_obs) #, object_masks=object_masks)
            return action, param #, is_valids, inds

    def add_observation(self, action, add_obs=True):
        self.step_count += 1
        self.explorer.add_observation(self.obs, action, add_obs=add_obs)


    def get_path_to_goal(self):
        return self.explorer.get_path_to_goal()

    def get_map_pos_from_aithor_pos(self, aithor_pos):
        return self.explorer.mapper.get_position_on_map_from_aithor_position(aithor_pos)

    def get_clostest_reachable_map_pos(self, map_pos):
        reachable = self.get_reachable_map_locations(sample=False)
        inds_i, inds_j = np.where(reachable)
        reachable_where = np.stack([inds_i, inds_j], axis=0)
        dist = distance.cdist(np.expand_dims(map_pos, axis=0), reachable_where.T)
        argmin = np.argmin(dist)
        ind_i, ind_j = inds_i[argmin], inds_j[argmin]
        return ind_i, ind_j

    def get_reachable_map_locations(self, sample=True):
        reachable = self.explorer._get_reachable_area()
        state_xy = self.explorer.mapper.get_position_on_map()
        if sample:
            inds_i, inds_j = np.where(reachable)
            dist = np.sqrt(np.sum(np.square(np.expand_dims(state_xy,axis=1) - np.stack([inds_i, inds_j], axis=0)),axis=0))
            dist_thresh = dist>20.0
            inds_i = inds_i[dist_thresh]
            inds_j = inds_j[dist_thresh]
            if inds_i.shape[0]==0:
                print("FOUND NO REACHABLE INDICES")
                return None, None
            ind = np.random.randint(inds_i.shape[0])
            ind_i, ind_j = inds_i[ind], inds_j[ind]

            return ind_i, ind_j
        else:
            return reachable

    def get_interaction_reachable_map_pos(self, object_center, location_quandrant='first', nbins = 8):
        '''
        Gets navigable points around the agent at various yaw angles (bins navigable points around the agent based on yaw angle around object center)
        object_center: object center in map coordinates
        location_quandrant: center is closest point, right is right quandrant to cloest point, etc.
        '''

        if location_quandrant=='first':
            return self.get_clostest_reachable_map_pos(object_center)
        elif location_quandrant in ['second', 'third']:
            map_step_resolution = int(self.explorer.STEP_SIZE/self.explorer.mapper.resolution)
            closest_point = np.array(self.get_clostest_reachable_map_pos(object_center))
            valid_pts = np.asarray(np.where(self.get_reachable_map_locations(sample=False)))

            closest_point_res = map_step_resolution * np.round(closest_point/map_step_resolution)
            where_too_close = np.linalg.norm(valid_pts - np.expand_dims(closest_point_res, axis=1), axis=0) < map_step_resolution*2
            
            where_too_close_ = np.linalg.norm(valid_pts - np.expand_dims(self.explorer.mapper.get_position_on_map(), axis=1), axis=0) < map_step_resolution*2
            
            where_too_close = np.logical_or(where_too_close, where_too_close_)

            valid_pts = valid_pts[:,~where_too_close]
            dist = distance.cdist(np.expand_dims(object_center, axis=0), valid_pts.T)
            if dist.size==0:
                return self.get_clostest_reachable_map_pos(object_center)
            argmin = np.argmin(dist)
            second_closest_point = valid_pts[:,argmin]

            if location_quandrant=="second":
                return second_closest_point

            where_too_close = np.linalg.norm(valid_pts - np.expand_dims(second_closest_point, axis=1), axis=0) < map_step_resolution*2

            valid_pts = valid_pts[:,~where_too_close]
            dist = distance.cdist(np.expand_dims(object_center, axis=0), valid_pts.T)
            if dist.size==0:
                return self.get_clostest_reachable_map_pos(object_center)
            argmin = np.argmin(dist)
            third_closest_point = valid_pts[:,argmin]                

            visualize = False
            if visualize:
                map_vis = self.get_reachable_map_locations(sample=False)
                plt.figure()
                plt.imshow(map_vis)
                plt.plot(valid_pts[1], valid_pts[0], 'x', color = 'magenta')
                plt.plot(object_center[1], object_center[0], 'o', color = 'black')
                plt.plot(closest_point[1], closest_point[0], 'x', color = 'blue')
                plt.plot(closest_point_res[1], closest_point_res[0], 'x', color = 'green')
                plt.plot(second_closest_point[1], second_closest_point[0], 'x', color = 'red')
                plt.plot(third_closest_point[1], third_closest_point[0], 'x', color = 'teal')
                pos_ = self.explorer.mapper.get_position_on_map()
                plt.plot(pos_[0], pos_[1], 'x', color = 'cyan')
                plt.savefig('output/images/test.png')

                plt.figure()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                plt.imshow(rgb)
                plt.savefig('output/images/test1.png')
                st()

            return third_closest_point


        elif False: 
            closest_point = np.array(self.get_clostest_reachable_map_pos(object_center))

            valid_pts = np.asarray(np.where(self.get_reachable_map_locations(sample=False)))

            valid_pts_shift = valid_pts - object_center[:,None]
            valid_pts_shift = valid_pts_shift.T

            dz = valid_pts_shift[:,1]
            dx = valid_pts_shift[:,0]

            # Get yaw for binning 
            valid_yaw = np.degrees(np.arctan2(dz,dx))

            # binning yaw around object
            bins = np.linspace(-180, 180, nbins+1)
            bin_yaw = np.digitize(valid_yaw, bins)

            num_valid_bins = np.unique(bin_yaw).size

            # get distance of object to each bin
            dists_object = []
            for bi in range(nbins):
                cur_bi = np.where(bin_yaw==(bi+1))[0]
                if len(cur_bi)==0:
                    dists_object.append(999999)
                    continue
                points = valid_pts[:,cur_bi]
                dists_object.append(np.min(np.linalg.norm(points - object_center[:,None], axis=0)))
            dists_object = np.array(dists_object)
            # bin_closest = np.argmin(dists)+1
            argsort_dist_object = np.argsort(dists_object)+1

            if location_quandrant=='third':
                bin_chosen_idx = argsort_dist_object[2]
                if dists_object[bin_chosen_idx-1]==999999:
                    location_quandrant = 'second'

            if location_quandrant=='second':
                bin_chosen_idx = argsort_dist_object[1]
                if dists_object[bin_chosen_idx-1]==999999:
                    bin_chosen_idx = argsort_dist_object[0]
            
            cur_bi = np.where(bin_yaw==bin_chosen_idx)[0]
            points = valid_pts[:,cur_bi]
            min_idx = np.argmin(np.linalg.norm(points - object_center[:,None], axis=0))
            chosen_point = points[:,min_idx]

            visualize = False
            if visualize:
                import matplotlib.cm as cm
                colors = iter(cm.rainbow(np.linspace(0, 1, nbins)))
                plt.figure(2)
                plt.clf()
                # print(np.unique(bin_yaw))
                for bi in range(nbins):
                    cur_bi = np.where(bin_yaw==(bi+1))[0]
                    if len(cur_bi)==0:
                        continue
                    points = valid_pts[:,cur_bi]
                    x_sample = points[0,:]
                    z_sample = points[1,:]
                    plt.plot(z_sample, x_sample, 'o', color = next(colors))
                cur_bi = np.where(bin_yaw==bin_chosen_idx)[0]
                points = valid_pts[:,cur_bi]
                x_sample = points[0,:]
                z_sample = points[1,:]
                plt.plot(z_sample, x_sample, 'x', color = 'orange')
                plt.plot(object_center[1], object_center[0], 'x', color = 'black')
                plt.plot(closest_point[1], closest_point[0], 'x', color = 'blue')
                plt.plot(chosen_point[1], chosen_point[0], 'x', color = 'red')
                plt.savefig(f'data/images/test_{location_quandrant}.png')
                plt.close()
                st()

            return list(chosen_point)

        else:
            assert(False) # what location_quandrant is this?


    def set_point_goal(self, ind_i, ind_j, dist_thresh=0.3, search_mode=False, search_mode_reduced=False):
        '''
        ind_i and ind_j are indices in the map
        we denote camX0 as the first camera angle + position (spawning position) and camX as current camera angle + position
        '''
        print(f"Setting point goal to {ind_i}, {ind_j}")
        self.obs.goal = Namespace(metadata={"category": 'point_nav', "targets":[ind_i, ind_j]})
        self.parse_goal(self.obs.goal)
        self.explorer.set_point_goal(ind_i, ind_j, dist_thresh=dist_thresh, search_mode=search_mode, search_mode_reduced=search_mode_reduced)

    def search_random_locs_for_object(
        self, 
        search_object,
        max_steps=75, 
        vis=None, 
        text='', 
        object_tracker=None, 
        max_fail=30, 
        search_mode=False,
        search_mode_reduced=False,
        num_search_locs_object=args.num_search_locs_object,
        ):
        '''
        Search random goals for object
        search_object: object category string or list of category strings - stop navigation and return object info when found
        '''

        start_steps = self.task.steps

        for n in range(num_search_locs_object):
            ind_i, ind_j = self.get_reachable_map_locations(sample=True)
            if ind_i is None:
                return {}
            self.set_point_goal(ind_i, ind_j, dist_thresh=args.dist_thresh, search_mode=search_mode, search_mode_reduced=search_mode_reduced)

            out = self.navigate_to_point_goal(
                vis=vis, 
                text=f"Search for {search_object}", 
                object_tracker=object_tracker,
                search_object=search_object,
                )

            if (self.task.steps - start_steps) >= max_steps:
                return {}

            if len(out)>0:
                return out

        return out

    def search_local_region(
        self, 
        vis=None, 
        text='Search local region', 
        object_tracker=None, 
        search_object=None,
        search_object_in_view=False, 
        map_pos=None,
        ):
        '''
        Search local region around agent where it is standing
        search_object: object category string or list of category strings - stop navigation and return object info when found
        '''

        out = {}

        action_sequence = ['RotateLeft', 'LookUp', 'RotateRight', 'RotateRight', 'LookDown', 'LookDown', 'RotateLeft', 'RotateLeft', 'LookUp', 'RotateRight']
        
        # self.bring_head_to_center(vis=vis)

        steps = 0
        found_obj = False
        num_failed = 0

        for search_loc_i in range(2): # try in two locations
            print(F"Local search in location #{search_loc_i}")
            if search_loc_i>0:
                if map_pos is None:
                    continue
                quadrants = ["second", "third"]
                ind_i, ind_j  = self.get_interaction_reachable_map_pos(map_pos, location_quandrant=quadrants[search_loc_i-1])
                self.set_point_goal(ind_i, ind_j, dist_thresh=args.dist_thresh) # set point goal in map
                self.navigate_to_point_goal(vis=vis, text=f"Navigate to {search_object}", object_tracker=object_tracker, max_fail=3, add_obs=True)
            for action in action_sequence:

                if action=="LookDown" and self.explorer.head_tilt==60:
                    continue

                if action=="LookUp" and self.explorer.head_tilt==-30:
                    continue

                if self.task.is_done():
                    print("Task done! Skipping search.")
                    break

                if args.verbose:
                    print(f"search_local_region: {action}")

                self.task.step(action=action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb,depth, action_successful)
                # whenever not acting - add obs
                self.add_observation(action, add_obs=False)

                camX0_T_camX = self.explorer.get_camX0_T_camX()

                if vis is not None:
                    vis.add_frame(rgb, text=text)

                if object_tracker is not None and action_successful:
                    object_tracker.update(rgb, depth, camX0_T_camX, vis=vis, target_object=search_object, only_keep_target=True if search_object is not None else False)

                    if search_object is not None:
                        if search_object_in_view:
                            interact_mask = object_tracker.get_predicted_masks(
                                                        rgb, 
                                                        object_category=search_object, 
                                                        score_threshold=0.0,
                                                        max_masks=3,
                                                        )
                            if len(interact_mask)>0:
                                return interact_mask
                        else:
                            centroids, labels, object_ids = object_tracker.get_centroids_and_labels(
                                return_ids=True, object_cat=search_object
                                )
                            if len(centroids)>0:
                                out = {'centroids':centroids, 'labels':labels, 'object_ids':object_ids}
                                return out
        if search_object is not None:
            return {}           

    def navigate_to_point_goal(
        self, 
        max_steps=75, 
        vis=None, 
        text='', 
        object_tracker=None, 
        search_object=None,
        max_fail=30, 
        add_obs=False,
        ):
        '''
        search_object: object category string or list of category strings - stop navigation and return object info when found
        '''


        steps = 0
        num_failed = 0
        while True:

            if self.task.is_done():
                print("Task done! Skipping navigation to point goal.")
                break

            action, param = self.act(add_obs=False)
            
            camX0_T_camX = self.explorer.get_camX0_T_camX()

            if steps>0:
                if vis is not None:
                    vis.add_frame(rgb, text=text, add_map=True)
                if object_tracker is not None and action_successful:
                    object_tracker.update(rgb, depth, camX0_T_camX, vis=vis, target_object=search_object, only_keep_target=True if search_object is not None else False)

            if args.verbose:
                print(f"navigate_to_point_goal: {action}") #, action_rearrange, action_ind)

            if action=='Pass':
                print("Pass reached.")
                if steps==0:
                    rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                    if vis is not None:
                        vis.add_frame(rgb, text=text, add_map=True)
                break
            else:
                self.task.step(action=action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb, depth, action_successful)

            if not action_successful:
                num_failed += 1

            steps += 1

            if object_tracker is not None:
                if search_object is not None:
                    centroids, labels, object_ids = object_tracker.get_centroids_and_labels(
                        return_ids=True, object_cat=search_object
                        )
                    if len(centroids)>0:
                        out = {'centroids':centroids, 'labels':labels, 'object_ids':object_ids}
                        _, _ = self.act(add_obs=False, action='pass') # need this to make sure previous action was added in explore.py
                        return out

            if steps > max_steps:
                _, _ = self.act(add_obs=False, action='pass') # need this to make sure previous action was added in explore.py
                break

            if max_fail is not None:
                if num_failed >= max_fail:
                    if args.verbose: 
                        print("Max fail reached.")
                    _, _ = self.act(add_obs=False, action='pass') # need this to make sure previous action was added in explore.py
                    break

        if search_object is not None:
            return {}   
        

    def orient_camera_to_point(
        self, 
        target_position, 
        vis=None, 
        text='Orient to object', 
        object_tracker=None
        ):


        pos_s = self.explorer.get_agent_position_camX0()
        target_position = np.array(list(target_position.values()))

        # YAW calculation - rotate to object
        agent_to_obj = np.squeeze(target_position) - pos_s 
        agent_local_forward = np.array([0, 0, 1.0]) 
        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
        flat_to_obj /= flat_dist_to_obj

        det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

        turn_yaw = np.degrees(turn_angle) 
        turn_pitch = np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))

        yaw_cur, pitch_cur, _ = self.explorer.get_agent_rotations_camX0()

        relative_yaw = yaw_cur - turn_yaw
        if relative_yaw<-180:
            relative_yaw = relative_yaw+360
        if relative_yaw>180:
            relative_yaw = relative_yaw-360
        if relative_yaw > 0:
            yaw_action = 'RotateLeft'
        elif relative_yaw < 0:
            yaw_action = 'RotateRight'
        num_yaw = int(np.nan_to_num(np.abs(np.round(relative_yaw / args.DT))))

        if num_yaw > 0:
            for t in range(num_yaw):

                if self.task.is_done():
                    print("Task done! Skipping orient to point goal.")
                    break

                if args.verbose:
                    print(f"orient_camera_to_point: {yaw_action}")

                self.task.step(action=yaw_action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb,depth, action_successful)
                # whenever not acting - add obs
                self.add_observation(yaw_action, add_obs=False)

                if vis is not None:
                    vis.add_frame(rgb, text=text)

                if object_tracker is not None and action_successful:
                    camX0_T_camX = self.explorer.get_camX0_T_camX()
                    object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

        if turn_pitch > 30.:
            turn_pitch = 30.
        if turn_pitch < -60.:
            turn_pitch = -60.

        relative_pitch = turn_pitch - pitch_cur
        if relative_pitch < 0:
            pitch_action = 'LookDown'
        elif relative_pitch > 0:
            pitch_action = 'LookUp'
        else:
            pitch_action = None
        num_pitch = int(np.nan_to_num(np.abs(np.round(relative_pitch / args.HORIZON_DT))))
        print("num_pitch",num_pitch)

        if num_pitch > 0:
            for t in range(num_pitch):

                if self.task.is_done():
                    print("Task done! Skipping orient to point goal.")
                    break

                if args.verbose:
                    print(f"orient_camera_to_point: {pitch_action}")
                print(f"orient_camera_to_point: {pitch_action}")

                self.task.step(action=pitch_action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb,depth, action_successful)
                # whenever not acting - add obs
                self.add_observation(pitch_action, add_obs=False)

                if vis is not None:
                    vis.add_frame(rgb, text=text)

                if object_tracker is not None and action_successful:
                    camX0_T_camX = self.explorer.get_camX0_T_camX()
                    object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

    def step_back(
        self, 
        vis=None, 
        text='Step back', 
        object_tracker=None,
        ):

        rotate_action = 'RotateLeft'
        rotate_action_inv = 'RotateRight'

        num = 2
        for t in range(num):

            if self.task.is_done():
                break

            self.task.step(action=rotate_action)

            if t==0 and not self.task.action_success():
                # rotate the other way
                rotate_action = 'RotateRight'
                rotate_action_inv = 'RotateLeft'
                self.task.step(action=rotate_action)

            action_successful = self.task.action_success()
            rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
            self.update_navigation_obs(rgb,depth, action_successful)
            # whenever not acting - add obs
            self.add_observation(rotate_action, add_obs=False)

            if vis is not None:
                vis.add_frame(rgb, text=text)

            if object_tracker is not None and action_successful:
                camX0_T_camX = self.explorer.get_camX0_T_camX()
                object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

        num = 1
        for t in range(num):

            if self.task.is_done():
                break

            self.task.step(action='MoveAhead')

            action_successful = self.task.action_success()
            rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
            self.update_navigation_obs(rgb,depth, action_successful)
            # whenever not acting - add obs
            self.add_observation('MoveAhead', add_obs=False)

            if vis is not None:
                vis.add_frame(rgb, text=text)

            if object_tracker is not None and action_successful:
                camX0_T_camX = self.explorer.get_camX0_T_camX()
                object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

        num = 2
        for t in range(num):

            if self.task.is_done():
                break

            self.task.step(action=rotate_action_inv)

            action_successful = self.task.action_success()
            rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
            self.update_navigation_obs(rgb,depth, action_successful)
            # whenever not acting - add obs
            self.add_observation(rotate_action_inv, add_obs=False)

            if vis is not None:
                vis.add_frame(rgb, text=text)

            if object_tracker is not None and action_successful:
                camX0_T_camX = self.explorer.get_camX0_T_camX()
                object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

    def adjust_depth(self, depth_frame, is_holding=False):
        # plt.figure()
        # plt.imshow(depth_frame)
        # plt.colorbar()
        # plt.savefig('output/images/test.png')
        # mask_err_below = depth_frame < 0.5
        # depth_frame[mask_err_below] = np.nan
        if is_holding:
            mask_err_below = depth_frame < 0.6
            depth_frame[mask_err_below] = np.nan
        return depth_frame


    def explore_env(
        self, 
        vis=None, 
        object_tracker=None, 
        max_fail=None, 
        max_steps=200, 
        return_obs_dict=False,
        use_aithor_coord_frame=False,
        close_walls_after_explore=False,
        ):
        '''
        This function explores the environment based on 
        '''
        
        fig = None 

        step = 0
        valid = 0
        num_failed = 0
        if return_obs_dict:
            obs_dict = {'rgb':[], 'xyz':[], 'camX0_T_camX':[], 'camX0_candidates':[], 'pitch':[], 'yaw':[]}
        change_list = []
        while True:
            print("exploring")

            if step==0:
                
                self.init_navigation(None)

                if object_tracker is not None:
                    object_tracker.navigation = self

                if use_aithor_coord_frame:
                    camX0_T_camX = utils.aithor.get_origin_T_camX(self.task.controller.last_event, True)
                else:
                    camX0_T_camX = self.explorer.get_camX0_T_camX()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                action_successful = True

                self.mapping_images.append(rgb)

                self.update_navigation_obs(rgb,depth, action_successful)

                if use_aithor_coord_frame:
                    camX0_T_camX = utils.aithor.get_origin_T_camX(self.task.controller.last_event, True)
                else:
                    camX0_T_camX = self.explorer.get_camX0_T_camX()
                camX0_T_camX0 = utils.geom.safe_inverse_single(camX0_T_camX)
                if vis is not None:
                    vis.camX0_T_camX0 = camX0_T_camX0
                    if object_tracker is not None:
                        vis.object_tracker = object_tracker

                if return_obs_dict:
                    # need this for supervision for visual search network (not used during inference)
                    origin_T_camX0_invert = utils.aithor.get_origin_T_camX(self.task.controller.last_event, True)

            action, param = self.act()
            action_ind = self.action_mapping[action]

            print(action)

            if use_aithor_coord_frame:
                camX0_T_camX = utils.aithor.get_origin_T_camX(self.task.controller.last_event, True)
            else:
                camX0_T_camX = self.explorer.get_camX0_T_camX()

            if return_obs_dict:
                rgb_x = rgb/255.
                rgb_x = torch.from_numpy(rgb_x).permute(2,0,1).float()
                depth_ = torch.from_numpy(depth).cuda().unsqueeze(0).unsqueeze(0)
                xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).float())
                depth_threshold = 0.5
                percent_depth_thresh = 0.5 # only keep views with proportion of depth > threshold (dont want looking at wall)
                if (np.count_nonzero(depth<depth_threshold)/(args.H*args.W) < percent_depth_thresh):
                    obs_dict['rgb'].append(rgb_x.unsqueeze(0))
                    obs_dict['xyz'].append(xyz)
                    obs_dict['camX0_T_camX'].append(camX0_T_camX.unsqueeze(0))
                    obs_dict['pitch'].append(torch.tensor([self.explorer.head_tilt]))
                    obs_dict['yaw'].append(torch.tensor([self.explorer.rotation]))
                    is_camX0_candidate = np.round(self.explorer.head_tilt)==0 and np.round(self.explorer.rotation)%90==0.
                    if is_camX0_candidate:
                        obs_dict['camX0_candidates'].append(torch.tensor([valid]))
                    valid += 1

            if vis is not None:
                vis.add_frame(rgb, text="Explore", add_map=True)

            if object_tracker is not None and action_successful:
                object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)
            

            if args.verbose:
                print(f"explore_env: {action}")

            if action=='Pass':
                break
            else:
                self.task.step(action=action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb,depth, action_successful)

                self.mapping_images.append(rgb)

            if not action_successful:
                num_failed += 1
            if max_fail is not None:
                if num_failed >= max_fail:
                    if args.verbose: 
                        print("Max fail reached.")
                    _, _ = self.act(add_obs=False, action='pass') # need this to make sure previous action was added in explore.py
                    break
            if step==max_steps:
                _, _ = self.act(add_obs=False, action='pass') # need this to make sure previous action was added in explore.py
                break

            step += 1

        print("num steps taken:", step)

        if close_walls_after_explore:
            self.explorer.mapper.force_closed_walls()

        if vis is not None:
            for _ in range(5):
                vis.add_frame(rgb, text="Done Explore", add_map=True)

        if return_obs_dict:
            for key in list(obs_dict.keys()):
                obs_dict[key] = torch.cat(obs_dict[key], dim=0).cpu().numpy()
            if object_tracker is not None:
                obs_dict["objects_track_dict"] = object_tracker.objects_track_dict
            obs_dict['origin_T_camX0'] = origin_T_camX0_invert.cpu().numpy()
            return obs_dict

    def get_obs(self, head_tilt=None, return_sem_seg=False, adjust_depth=True):
        obs = self.task.get_observations()
        rgb = obs["rgb"]
        sem_seg_pred = None
        if args.use_estimated_depth:
            depth, sem_seg_pred = self.depth_estimator.get_depth_map(rgb, head_tilt)
            # plt.figure()
            # plt.imshow(depth)
            # plt.colorbar()
            # plt.savefig('output/images/test.png')
        else:
            depth = obs["depth"]
        if adjust_depth:
            depth = self.adjust_depth(depth.copy(), True if "is_holding" not in obs.keys() else obs["is_holding"])
        if return_sem_seg:
            return rgb, depth, sem_seg_pred
        return rgb, depth

    def update_navigation_obs(self, rgb, depth, action_successful):
        '''
        updates navigation mapping inputs
        rgb: rgb of current frame
        depth: depth map current frame
        action_successful: whether previous action was successful
        navigation: navigation class 
        '''
        # if adjust_depth:
        #     obs = self.task.get_observations()
        #     depth = self.adjust_depth(depth.copy(), True if "is_holding" not in obs.keys() else obs["is_holding"])
        self.obs.image_list = [rgb]
        self.obs.depth_map_list = [depth]
        self.obs.return_status = "SUCCESSFUL" if action_successful else "OBSTRUCTED"

    def bring_head_to_angle(
        self, 
        vis=None,
        text='head to angle',
        object_tracker=None, 
        update_obs=True, 
        angle=0,
        ):

        pitch_cur = self.task.get_agent_head_tilt() #self.explorer.get_agent_rotations_camX0()

        relative_pitch = angle - pitch_cur
        if relative_pitch < 0:
            pitch_action = 'LookUp'
        elif relative_pitch > 0:
            pitch_action = 'LookDown'
        else:
            pitch_action = None
        num_pitch = int(np.abs(np.round(relative_pitch / args.HORIZON_DT)))

        if num_pitch > 0:
            for t in range(num_pitch):

                print(f"bring_head_to_center: {pitch_action}")

                if object_tracker is not None and t>0 and action_successful:
                    camX0_T_camX = self.explorer.get_camX0_T_camX()
                    object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)
                
                self.task.step(action=pitch_action)

                if update_obs:
                    action_successful = self.task.action_success()
                    rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                    self.update_navigation_obs(rgb,depth, action_successful)
                    # whenever not acting - add obs
                    self.add_observation(pitch_action, add_obs=False)

                    if vis is not None:
                        vis.add_frame(rgb, text=text)

    def take_action(
        self, 
        action,
        vis=None,
        text='head to angle',
        object_tracker=None, 
        ):

        if self.task.is_done():
            return

        self.task.step(action=action)

        action_successful = self.task.action_success()
        rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
        self.update_navigation_obs(rgb,depth, action_successful)
        # whenever not acting - add obs
        self.add_observation(action, add_obs=False)

        if vis is not None:
            vis.add_frame(rgb, text=text)

        if object_tracker is not None and action_successful:
            camX0_T_camX = self.explorer.get_camX0_T_camX()
            object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

    def update_obs(
        self, 
        rgb, 
        depth, 
        action_successful, 
        update_success_checker=False
        ):
        self.obs.image_list = [rgb]
        self.obs.depth_map_list = [depth]
        self.obs.return_status = "SUCCESSFUL" if action_successful else "OBSTRUCTED"
        if update_success_checker:
            self.success_checker.update_image(rgb) # update action success checker with new image

class CheckSuccessfulAction():
    def __init__(self, rgb_init, H, W, perc_diff_thresh = 0.01):
        '''
        rgb_init: the rgb image from the spawn viewpoint W, H, 3
        This class does a simple check with the previous image to see if it completed the action 
        '''
        self.rgb_prev = rgb_init
        self.perc_diff_thresh = perc_diff_thresh
        self.H = H
        self.W = W

    def update_image(self, rgb):
        self.rgb_prev = rgb

    def check_successful_action(self, rgb, action):
        wheres = np.where(self.rgb_prev != rgb)
        wheres_ar = np.zeros(self.rgb_prev.shape)
        wheres_ar[wheres] = 1
        wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
        connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
        unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
        max_area = -1
        for lab in unique_labels:
            wheres_lab = np.where(connected_regions == lab)
            max_area = max(len(wheres_lab[0]), max_area)
        if (action in ['OpenObject', 'CloseObject']) and max_area > 500:
            success = True
        elif max_area > 100:
            success = True
        else:
            success = False
        return success

    
if __name__ == '__main__':
    pass