import logging
import numpy as np
from ast import literal_eval
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.gridspec as gridspec
from numpy import ma
import scipy, skfmm
from .mapper import Mapper
from .depth_utils import get_camera_matrix
from .fmm_planner import FMMPlanner
from .shortest_path_planner import ShortestPathPlanner
import skimage
from skimage.measure import label  
from textwrap import wrap
from queue import LifoQueue 
from matplotlib import pyplot as plt 
import math
import cv2
import torch
import copy

import ipdb
st = ipdb.set_trace
from arguments import args

import matplotlib.pyplot as plt
import tkinter
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.spatial import distance
do_video = False
video_name = 'images/output.avi'
use_cuda = True
use_rgb = True
######## Detectron2 imports end ########

STOP = 0
FORWARD = 3
BACKWARD = 15
LEFTWARD = 16
RIGHTWARD = 17
LEFT = 2
RIGHT = 1
UNREACHABLE = 5
EXPLORED = 6
DONE = 7
DOWN = 8
PICKUP = 9
OPEN = 10
PUT = 11
DROP = 12
UP = 13
CLOSE = 14

POINT_COUNT = 1
actions = { 
            # STOP: 'RotateLook', 
            STOP: 'MoveAhead', 
            LEFT: 'RotateLeft', 
            RIGHT: 'RotateRight', 
            FORWARD: 'MoveAhead',
            BACKWARD: 'MoveBack',
            LEFTWARD: 'MoveLeft',
            RIGHTWARD: 'MoveRight',
            DONE: 'Pass',
            DOWN: 'LookDown',
            UP: 'LookUp',
            PICKUP: 'PickupObject',
            OPEN: 'OpenObject',
            CLOSE: 'CloseObject',
            PUT: 'PutObject',
            DROP: 'DropObject',
          }
actions_inv = { 
          'RotateLeft': LEFT, 
          'RotateRight': RIGHT, 
          'MoveAhead': FORWARD,
          'MoveBack': BACKWARD,
          'LookDown': DOWN,
          'LookUp': UP,
          'MoveLeft':LEFTWARD,
          'MoveRight':RIGHTWARD,
          'MoveBack':BACKWARD,
          'pass': DONE,
        }

class Explore():
    def __init__(
        self, 
        obs, 
        goal, 
        bounds, z=[0.15, 2.0], 
        keep_head_down=False, 
        keep_head_straight=False, 
        look_down_init=True,
        dist_thresh=0.5, 
        search_pitch_explore=False, 
        block_map_positions_if_fail=False,
        use_FMM_planner=False
        ):

        self.params = { 
           STOP: {'degrees': 0}, 
           LEFT: {'degrees': obs.DT}, 
           RIGHT: {'degrees': obs.DT}, 
           FORWARD: {'moveMagnitude': obs.STEP_SIZE},
           BACKWARD: {'moveMagnitude': obs.STEP_SIZE},
           LEFTWARD: {'moveMagnitude': obs.STEP_SIZE},
           RIGHTWARD: {'moveMagnitude': obs.STEP_SIZE},
           DONE: {},
           DOWN: {'degrees': obs.HORIZON_DT},
           UP: {'degrees': -obs.HORIZON_DT},
           PICKUP: {'objectId': None},
           OPEN: {'objectId': None, 'amount': 0.99},
           CLOSE: {'objectId': None, 'amount': 0.99},
           PUT: {'objectId': None, 'receptacleObjectId': None},
           DROP: {'objectId': None},
        }

        self.actions_inv = { 
          'RotateLeft': LEFT, 
          'RotateRight': RIGHT, 
          'MoveAhead': FORWARD,
          'MoveBack': BACKWARD,
          'LookDown': DOWN,
          'LookUp': UP,
          'pass': DONE,
        }

        self.actions = actions

        self.DT = obs.DT
        self.STEP_SIZE = obs.STEP_SIZE
        self.HORIZON_DT = obs.HORIZON_DT
        self.head_tilt = obs.head_tilt_init 
        print(self.STEP_SIZE)
        print(self.HORIZON_DT)
        print(self.DT)
        print(self.head_tilt)

        self.keep_head_down = keep_head_down
        self.keep_head_straight = keep_head_straight
        if self.HORIZON_DT==30:
            self.num_down_explore = 1
            self.num_down_nav = 1
            self.init_down = 2
        elif self.HORIZON_DT==15:
            self.num_down_explore = 3
            self.num_down_nav = 3
            self.init_down = 3
        else:
            assert(False)
        
        self.max_steps_coverage = 1000
        self.max_steps_pointnav = 100
        self.max_steps_pointnav_cover = 20
        self.do_init_down = False
        self.init_on = True
        self.search_pitch_explore = False #search_pitch_explore
        self.look_down_init = look_down_init

        self.use_FMM_planner = use_FMM_planner

        self.do_visualize = False
        self.step_count = 0
        self.goal = goal
        
        self.rng = np.random.RandomState(0)
        self.fmm_dist = np.zeros((1,1))
        self.acts = iter(())
        self.acts_og = iter(())
        self.explored = False
        self.map_size = 12 #17 # max scene bounds for Ai2thor is ~11.5 #12 # 25
        self.resolution = 0.05
        self.max_depth = 200. # 4. * 255/25.
        self.dist_thresh = dist_thresh # initial dist_thresh
        if self.use_FMM_planner:
            self.add_obstacle_if_action_fail = True
        else:
            self.add_obstacle_if_action_fail = False
        self.block_map_positions_if_fail = block_map_positions_if_fail
        self.explored_threshold = int(self.map_size/self.resolution*self.map_size/self.resolution*0.01)

        self.keep_head_down_init = False
        self.exploring_max_depth_image = None
        if self.keep_head_down: # estimated depth
            self.selem = skimage.morphology.disk(2) 
            self.max_depth_image = None 
            self.exploring_max_depth_image = None 
            if self.HORIZON_DT==30:
                self.view_angles = [30]
            elif self.HORIZON_DT==15:
                self.view_angles = [45]
            else:
                assert(False)
            loc_on_map_size = int(self.STEP_SIZE/self.resolution)   
            self.loc_on_map_selem = skimage.morphology.square(loc_on_map_size) 
        else:
            self.selem = skimage.morphology.square(int(np.ceil(0.23*1.5/self.resolution))) 
            self.max_depth_image = None
            self.view_angles = None
            loc_on_map_size = int((self.STEP_SIZE/self.resolution)*1.5) 
            self.loc_on_map_selem = skimage.morphology.square(loc_on_map_size) 
            if args.use_estimated_depth:
                self.max_depth_image = 1.5
                self.exploring_max_depth_image = 1.0
                self.num_down_explore = 1
                self.num_down_nav = 1
                self.init_down = 1
                self.keep_head_down_init = True
            if args.increased_explore:
                self.exploring_max_depth_image = 1.0
                self.keep_head_down = True

        self.unexplored_area = np.inf
        self.next_target = 0
        self.opened = []
        self._setup_execution(goal)

        # Default initial position (have camera height)
        self.position = {'x': 0, 
                         'y': obs.camera_height,
                         'z': 0}
        self.rotation = 0.
        self.prev_act_id = None
        self.obstructed_actions = []
        self.obstructed_states = []
        self.success = False
        self.point_goal = None

        self.z_bins = z #[0.15, 2.3] # 1.57 is roughly camera height
        print("ZBINS", self.z_bins)

        ar = obs.camera_aspect_ratio
        vfov = obs.camera_field_of_view*np.pi/180
        focal = ar[1]/(2*math.tan(vfov/2))
        fov = abs(2*math.atan(ar[0]/(2*focal))*180/np.pi)
        self.sc = 1. #255./25. #1 #57.13
        fov, h, w = fov, ar[1], ar[0]
        C = get_camera_matrix(w, h, fov=fov)
        
        self.bounds = bounds # aithor bounds
        self.mapper = Mapper(C, self.sc, self.position, self.map_size, self.resolution,
                                max_depth=self.max_depth, z_bins=self.z_bins,
                                loc_on_map_selem = self.loc_on_map_selem,
                                bounds=self.bounds)

        self.video_ind = 1

        self.invert_pitch = True # invert pitch when fetching roation matrix? 
        self.camX0_T_origin = self.get_camX0_T_camX(get_camX0_T_origin=True)
        self.camX0_T_origin = self.safe_inverse_single(self.camX0_T_origin)

        self.steps_since_previous_failure = 0
        self.failures_in_a_row = 0

        self.repeat_previous = 0

        '''
        # setup trophy detector
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        if not use_rgb:
            cfg.MODEL.PIXEL_MEAN = [100.0, 100.0, 100.0, 100.0]
            cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
        cfg.MODEL.WEIGHTS = retrieval_detector_path
        if use_cuda:
            cfg.MODEL.DEVICE='cuda'
        else:
            cfg.MODEL.DEVICE='cpu'
        cfg.DATASETS.TEST = ("val",) 
        thing_classes = ['trophy', 'box_closed', 'box_opened']
        d = "train"
        DatasetCatalog.register("val", lambda d=d: val_dataset_function())
        MetadataCatalog.get("val").thing_classes = thing_classes
        self.trophy_cfg = cfg
        self.detector = DefaultPredictor(cfg)
        '''

    def get_traversible_map(self):
        return self.mapper.get_traversible_map(
                          self.selem, 1,loc_on_map_traversible=True)

    def get_explored_map(self):
        return self.mapper.get_explored_map(self.selem, 1)

    def get_map(self):
        return self.mapper.map

    def _setup_execution(self, goal):
        self.execution = LifoQueue(maxsize=200)
        if self.goal.category == 'point_nav':
            fn = lambda: self._point_goal_fn_assigned(np.array([self.goal.targets[0], self.goal.targets[1]]), explore_mode=False, iters=self.max_steps_pointnav)
            self.execution.put(fn)
        elif self.goal.category == 'cover':
            self.exploring = True
            print("EXPLORING")
            fn = lambda: self._cover_fn(self.goal.targets[0], self.max_steps_coverage, 1, 1)
            self.execution.put(fn)
        elif self.goal.category == 'retrieval':
            fn = lambda: self._retrieve_fn(self.goal.targets[0]) 
            self.execution.put(fn)
        elif self.goal.category == 'traversal':
            fn = lambda: self._traverse_fn(self.goal.targets[0], 1, 1) 
            self.execution.put(fn)
        elif self.goal.category == 'transferral-top':
            fn = lambda: self._put_fn(self.goal.targets[0], self.goal.targets[1]) 
            self.execution.put(fn)
            fn = lambda: self._traverse_fn(self.goal.targets[1], 1, 1) 
            self.execution.put(fn)
            fn = lambda: self._retrieve_fn(self.goal.targets[0]) 
            self.execution.put(fn)
        elif self.goal.category == 'transferral-next':
            fn = lambda: self._drop_fn()
            self.execution.put(fn)
            fn = lambda: self._traverse_fn(self.goal.targets[1], 1, 1) 
            self.execution.put(fn)
            fn = lambda: self._retrieve_fn(self.goal.targets[0]) 
            self.execution.put(fn)
        else:
            assert(False), f'Incorrext goal category: {self.goal.category}'
        
        
        fn = lambda: self._init_fn()
        self.execution.put(fn)
    
    # def _drop_fn(self):
    #     yield actions[DROP]

    '''
    # Andy: not needed for this task
    def _put_fn(self, uuid, to_uuid):
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            return

        yield actions[DOWN], {'horizon': STEEP_HORIZON_DT}
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            yield actions[DOWN], {'horizon': -STEEP_HORIZON_DT}
            return
        yield actions[DOWN], {'horizon': -STEEP_HORIZON_DT}

        yield actions[DOWN], {'horizon': HORIZON_DT}
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            yield actions[DOWN], {'horizon': -HORIZON_DT}
            return
        yield actions[DOWN], {'horizon': -HORIZON_DT}
        
        yield FORWARD
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            return
        yield actions[DOWN], {'horizon': HORIZON_DT}
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            yield actions[DOWN], {'horizon': -HORIZON_DT}
            return
        yield actions[DOWN], {'horizon': -HORIZON_DT}
    '''

    def _init_fn(self):

        self.init_on = True
        self.exploring = False

        if self.look_down_init:
            if self.keep_head_down:
                deg_up = min(60, self.init_down*self.HORIZON_DT) - self.head_tilt
            else:
                deg_up = 60 - self.head_tilt
            for i in range(int(deg_up/self.HORIZON_DT)):
                yield DOWN
        
        for i in range(int(360/self.DT)):
            yield LEFT

        if self.look_down_init and not self.keep_head_down_init:
            if self.keep_head_down:
                deg_up = self.head_tilt - min(60, self.num_down_nav*self.HORIZON_DT)
            for i in range(int(deg_up/self.HORIZON_DT)):
                yield UP

        self.init_on = False
        self.exploring = True
    
    def _get_action(self, ID):
        if type(ID) == int:
            return actions[ID], self.params[ID]
        else:
            return ID[0], ID[1]

    def set_point_goal(self, ind_i, ind_j, dist_thresh=0.5, explore_mode=False, search_mode=False, search_mode_reduced=False):
        self.exploring = False
        self.acts = iter(())
        self.acts_og = iter(())
        self.dists = []
        self.execution = LifoQueue(maxsize=200)
        self.point_goal = [ind_i, ind_j]
        self.dist_thresh = dist_thresh
        self.obstructed_actions = []
        # fn = lambda: self._point_goal_fn(np.array([ind_j, ind_i]), dist_thresh=dist_thresh, explore_mode=explore_mode)
        fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), dist_thresh=dist_thresh, explore_mode=False, search_mode=search_mode, search_mode_reduced=search_mode_reduced, iters=self.max_steps_pointnav)
        self.execution.put(fn)

    def add_observation(self, obs, action, add_obs=True):
        self.step_count += 1
        self.obs = obs
        self.prev_act_id = DONE # set to DONE since when we call act() again, we do not want another action added there

        act_id = actions_inv[action]

    
        if obs.return_status == 'SUCCESSFUL': # and self.prev_act_id is not None:
            self.steps_since_previous_failure += 1
            self.failures_in_a_row = 0
            if 'Rotate' in actions[act_id]:
                if 'Left' in actions[act_id]:
                    self.rotation -= self.params[act_id]['degrees']
                else:
                    self.rotation += self.params[act_id]['degrees']
                self.rotation %= 360
            elif 'Move' in actions[act_id]:
                if act_id == FORWARD:
                    self.position['x'] += np.sin(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                    self.position['z'] += np.cos(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                elif act_id == BACKWARD:
                    self.position['x'] -= np.sin(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                    self.position['z'] -= np.cos(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                elif act_id == LEFTWARD:
                    self.position['x'] -= np.cos(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                    self.position['z'] += np.sin(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                elif act_id == RIGHTWARD:
                    self.position['x'] += np.cos(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                    self.position['z'] -= np.sin(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
            elif 'Look' in actions[act_id]:
                self.head_tilt += self.params[act_id]['degrees']
        elif obs.return_status == 'OBSTRUCTED' and act_id is not None:
            self.steps_since_previous_failure = 0
            self.failures_in_a_row += 1
            print("ACTION FAILED.")
            prev_len = len(self.obstructed_actions)
            if prev_len>4000:
                pass
            else:
                if self.use_FMM_planner:
                    for idx in range(prev_len):
                        obstructed_acts = self.obstructed_actions[idx]
                        self.obstructed_actions.append(obstructed_acts+[act_id])
                    self.obstructed_actions.append([act_id])
                else:
                    if 'Move' in actions[act_id] or 'Rotate' in actions[act_id]:
                        obstructed_state = copy.deepcopy(self.position)
                        obstructed_rot = copy.deepcopy(self.rotation)
                        if 'Left' in actions[act_id]:
                            obstructed_rot %= 360
                            obstructed_rot -= self.params[act_id]['degrees']
                            act_id_ = FORWARD
                        elif 'Right' in actions[act_id]:
                            obstructed_rot += self.params[act_id]['degrees']
                            obstructed_rot %= 360
                            act_id_ = FORWARD
                        else:
                            act_id_ = act_id
                        if act_id_ == FORWARD:
                            obstructed_state['x'] += np.sin(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                            obstructed_state['z'] += np.cos(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                        elif act_id_ == BACKWARD:
                            obstructed_state['x'] -= np.sin(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                            obstructed_state['z'] -= np.cos(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                        elif act_id_ == LEFTWARD:
                            obstructed_state['x'] -= np.cos(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                            obstructed_state['z'] += np.sin(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                        elif act_id_ == RIGHTWARD:
                            obstructed_state['x'] += np.cos(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                            obstructed_state['z'] -= np.sin(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                        obstructed_position = np.array([obstructed_state['x'], obstructed_state['z']], np.float32)
                        obstructed_map_position = obstructed_position - self.mapper.origin_xz + self.mapper.origin_map*self.mapper.resolution
                        obstructed_map_position = obstructed_map_position / self.mapper.resolution
                        # obstructed_map_position = obstructed_map_position.astype(np.int32)
                        self.obstructed_states.append(obstructed_map_position)
                        if self.block_map_positions_if_fail:
                            self.mapper.remove_position_on_map(obstructed_state)
        return_status = obs.return_status
        # print("Step {0}, position {1} / {2}, rotation {3}".format(self.step_count, self.position, obs.position, self.rotation))
        rgb = np.array(obs.image_list[-1])
        depth = np.array(obs.depth_map_list[-1])

        self.mapper.add_observation(self.position, 
                                    self.rotation, 
                                    -self.head_tilt, 
                                    depth, add_obs=add_obs)
        if obs.return_status == 'OBSTRUCTED' and 'Move' in actions[act_id]: 
            if self.add_obstacle_if_action_fail:
                self.mapper.add_obstacle_in_front_of_agent(self.selem)
                if self.point_goal is not None:
                    
                    self.execution = LifoQueue(maxsize=200)
                    self.point_goal = self.get_clostest_reachable_map_pos(self.point_goal)
                    ind_i, ind_j = self.point_goal
                    fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), dist_thresh=self.dist_thresh, iters=self.max_steps_pointnav)
                    self.execution.put(fn)

    def get_agent_position_camX0(self):
        '''
        Get agent position in camX0 (first position) reference frame
        '''
        pos = torch.from_numpy(np.array(list(self.position.values())))
        pos = self.apply_4x4(self.camX0_T_origin.unsqueeze(0), pos.unsqueeze(0).unsqueeze(0)).squeeze().numpy()
        return pos

    def get_agent_rotations_camX0(self):
        '''
        Get agent rotation in camX0 (first position) reference frame
        '''
        yaw = self.rotation
        pitch = self.head_tilt
        if self.invert_pitch:
            pitch = -pitch
        roll = 0
        return yaw, pitch, roll

    def get_camX0_T_camX(self, get_camX0_T_origin=False):
        '''
        Get transformation matrix between first position (camX0) and current position (camX)
        '''
        position = np.array(list(self.position.values()))
        # position = position[[2,1,0]]
        # position[0] = position[0] # invert x
        # position[2] = position[2] # invert z
        # in aithor negative pitch is up - turn this on if need the reverse
        head_tilt = self.head_tilt
        if self.invert_pitch:
            head_tilt = -head_tilt
        rx = np.radians(head_tilt) #np.radians(event.metadata["agent"]["cameraHorizon"]) # pitch
        rotation = self.rotation
        if rotation >= 180:
            rotation = rotation - 360
        if rotation < -180:
            rotation = 360 + rotation
        ry = np.radians(rotation) #np.radians(rotation[1]) # yaw
        rz = 0. # roll is always 0
        rotm = self.eul2rotm_py(np.array([rx]), np.array([ry]), np.array([rz]))
        origin_T_camX = np.eye(4)
        origin_T_camX[0:3,0:3] = rotm
        origin_T_camX[0:3,3] = position
        origin_T_camX = torch.from_numpy(origin_T_camX)
        if get_camX0_T_origin:
            camX0_T_camX = origin_T_camX
        else:
            camX0_T_camX = torch.matmul(self.camX0_T_origin, origin_T_camX)
        return camX0_T_camX

    def safe_inverse_single(self, a):
        r, t = self.split_rt_single(a)
        t = t.view(3,1)
        r_transpose = r.t()
        inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
        bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
        inv = torch.cat([inv, bottom_row], 0)
        return inv

    def split_rt_single(self, rt):
        r = rt[:3, :3]
        t = rt[:3, 3].view(3)
        return r, t

    def apply_4x4(self, RT, xyz):
        B, N, _ = list(xyz.shape)
        ones = torch.ones_like(xyz[:,:,0:1])
        xyz1 = torch.cat([xyz, ones], 2)
        xyz1_t = torch.transpose(xyz1, 1, 2)
        # this is B x 4 x N
        xyz2_t = torch.matmul(RT, xyz1_t)
        xyz2 = torch.transpose(xyz2_t, 1, 2)
        xyz2 = xyz2[:,:,:3]
        return xyz2

    def eul2rotm_py(self, rx, ry, rz):
        # inputs are shaped B
        # this func is copied from matlab
        # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
        #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
        #        -sy            cy*sx             cy*cx]
        rx = rx[:,np.newaxis]
        ry = ry[:,np.newaxis]
        rz = rz[:,np.newaxis]
        # these are B x 1
        sinz = np.sin(rz)
        siny = np.sin(ry)
        sinx = np.sin(rx)
        cosz = np.cos(rz)
        cosy = np.cos(ry)
        cosx = np.cos(rx)
        r11 = cosy*cosz
        r12 = sinx*siny*cosz - cosx*sinz
        r13 = cosx*siny*cosz + sinx*sinz
        r21 = cosy*sinz
        r22 = sinx*siny*sinz + cosx*cosz
        r23 = cosx*siny*sinz - sinx*cosz
        r31 = -siny
        r32 = sinx*cosy
        r33 = cosx*cosy
        r1 = np.stack([r11,r12,r13],axis=2)
        r2 = np.stack([r21,r22,r23],axis=2)
        r3 = np.stack([r31,r32,r33],axis=2)
        r = np.concatenate([r1,r2,r3],axis=1)
        return r

        
    def act(self, obs, action=None, fig=None, point_goal=None, add_obs=True, object_masks=[], held_obj_depth=100.):
        # print("Exploring?", self.exploring)
        self.step_count += 1
        self.obs = obs
        if do_video:
            if self.step_count == 1:
                # self.video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'FMP4'), 4, (480,480))
                rand_int = np.random.randint(100)
                video_name = f'images/output{rand_int}.avi'
                self.video_writer = cv2.VideoWriter(video_name, 0, 4, (480,480))
            rgb = np.array(obs.image_list[-1]).astype(np.uint8)
            bgr = rgb[:,:,[2,1,0]]
            self.video_writer.write(bgr)
        if self.step_count == 1:
            ar = obs.camera_aspect_ratio
            vfov = obs.camera_field_of_view*np.pi/180
            focal = ar[1]/(2*math.tan(vfov/2))
            fov = abs(2*math.atan(ar[0]/(2*focal))*180/np.pi)
            # sc = 1. #255./25. #1 #57.13
            fov, h, w = fov, ar[1], ar[0]
            # map_size = 20 #12 # 25
            # resolution = 0.05
            # max_depth = 200. # 4. * 255/25.
            # max_depth = 5. * 255/25.
            C = get_camera_matrix(w, h, fov=fov)
            self.mapper = Mapper(C, self.sc, self.position, self.map_size, self.resolution,
                                 max_depth=self.max_depth, z_bins=self.z_bins,
                                 loc_on_map_selem = self.loc_on_map_selem,
                                 bounds=self.bounds)
                        
        else:
            if obs.return_status == 'SUCCESSFUL' and self.prev_act_id is not None:
                self.steps_since_previous_failure += 1
                self.failures_in_a_row = 0
                # self.obstructed_actions = []
                if 'Rotate' in actions[self.prev_act_id]:
                    if 'Left' in actions[self.prev_act_id]:
                        self.rotation -= self.params[self.prev_act_id]['degrees']
                    else:
                        self.rotation += self.params[self.prev_act_id]['degrees']
                    self.rotation %= 360
                elif 'Move' in actions[self.prev_act_id]:
                    if self.prev_act_id == FORWARD:
                        self.position['x'] += np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                        self.position['z'] += np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                    elif self.prev_act_id == BACKWARD:
                        self.position['x'] -= np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                        self.position['z'] -= np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                    elif self.prev_act_id == LEFTWARD:
                        self.position['x'] -= np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                        self.position['z'] += np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                    elif self.prev_act_id == RIGHTWARD:
                        self.position['x'] += np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                        self.position['z'] -= np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                elif 'Look' in actions[self.prev_act_id]:
                    self.head_tilt += self.params[self.prev_act_id]['degrees']

            elif obs.return_status == 'OBSTRUCTED' and self.prev_act_id is not None:
                self.steps_since_previous_failure = 0
                self.failures_in_a_row += 1
                print("ACTION FAILED.")
                prev_len = len(self.obstructed_actions)
                if prev_len>4000:
                    pass
                else:
                    if self.use_FMM_planner:
                        # print(prev_len)
                        for idx in range(prev_len):
                            obstructed_acts = self.obstructed_actions[idx]
                            self.obstructed_actions.append(obstructed_acts+[self.prev_act_id])
                        self.obstructed_actions.append([self.prev_act_id])
                    else:
                        if 'Move' in actions[self.prev_act_id] or 'Rotate' in actions[self.prev_act_id]:
                            obstructed_state = copy.deepcopy(self.position)
                            obstructed_rot = copy.deepcopy(self.rotation)
                            if 'Left' in actions[self.prev_act_id]:
                                obstructed_rot %= 360
                                obstructed_rot -= self.params[self.prev_act_id]['degrees']
                                act_id_ = FORWARD
                            elif 'Right' in actions[self.prev_act_id]:
                                obstructed_rot += self.params[self.prev_act_id]['degrees']
                                obstructed_rot %= 360
                                act_id_ = FORWARD
                            else:
                                act_id_ = self.prev_act_id
                            if act_id_ == FORWARD:
                                obstructed_state['x'] += np.sin(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                                obstructed_state['z'] += np.cos(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                            elif act_id_ == BACKWARD:
                                obstructed_state['x'] -= np.sin(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                                obstructed_state['z'] -= np.cos(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                            elif act_id_ == LEFTWARD:
                                obstructed_state['x'] -= np.cos(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                                obstructed_state['z'] += np.sin(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                            elif act_id_ == RIGHTWARD:
                                obstructed_state['x'] += np.cos(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                                obstructed_state['z'] -= np.sin(obstructed_rot/180*np.pi)*self.params[act_id_]['moveMagnitude']
                            obstructed_position = np.array([obstructed_state['x'], obstructed_state['z']], np.float32)
                            obstructed_map_position = obstructed_position - self.mapper.origin_xz + self.mapper.origin_map*self.mapper.resolution
                            obstructed_map_position = obstructed_map_position / self.mapper.resolution
                            # obstructed_map_position = obstructed_map_position.astype(np.int32)
                            self.obstructed_states.append(obstructed_map_position)
                            # self.obstructed_states
                            if self.block_map_positions_if_fail:
                                self.mapper.remove_position_on_map(obstructed_state)

        # head_tilt = obs.head_tilt
        return_status = obs.return_status
        # print("Step {0}, position {1} / {2}, rotation {3}".format(self.step_count, self.position, obs.position, self.rotation))
        rgb = np.array(obs.image_list[-1])
        depth = np.array(obs.depth_map_list[-1])

        if self.max_depth_image is not None or self.exploring_max_depth_image is not None:
            if self.exploring and self.exploring_max_depth_image is not None:
                depth[depth>self.exploring_max_depth_image] = np.nan
            elif self.max_depth_image is not None:
                depth[depth>self.max_depth_image] = np.nan

        if self.view_angles is not None:
            add_obs = False
            for vi, va in enumerate(self.view_angles):
                if abs(va - self.head_tilt) <= 5:
                    add_obs = True


        self.mapper.add_observation(self.position, 
                                    self.rotation, 
                                    -self.head_tilt, 
                                    depth,
                                    add_obs=add_obs)
        if obs.return_status == 'OBSTRUCTED' and 'Move' in actions[self.prev_act_id]: # and self.prev_act_id is not None:
            # print("Ohhhhoooooonnooooooooooooooooonnononononooo")
            # print("ACTION FAILED.")
            if self.add_obstacle_if_action_fail:
                self.mapper.add_obstacle_in_front_of_agent(self.selem)
                if self.point_goal is not None:
                    
                    # self.mapper.dilate_obstacles_around_agent(self.selem)
                    self.execution = LifoQueue(maxsize=200)
                    # recompute closest navigable point
                    self.point_goal = self.get_clostest_reachable_map_pos(self.point_goal)
                    ind_i, ind_j = self.point_goal
                    # fn = lambda: self._point_goal_fn(np.array([ind_j, ind_i]), dist_thresh=dist_thresh, explore_mode=explore_mode)
                    fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), dist_thresh=self.dist_thresh, iters=self.max_steps_pointnav)
                    self.execution.put(fn)
        
        if False:
            act_id = DONE
        else:
            eps = 5
            if action is not None:
                act_id = actions_inv[action]
            elif self.repeat_previous>0:
                act_id = self.prev_act_id
                self.repeat_previous -= 1
            elif obs.return_status == 'OBSTRUCTED' and len(self.obstructed_states)>=2 and len(set([tuple(a) for a in self.obstructed_states[-2:]]))==1 and self.prev_act_id in [RIGHT,LEFT]:
                self.repeat_previous = 2 # turn the other way
                if self.prev_act_id==actions_inv['RotateLeft']:
                    act_id = RIGHT
                elif self.prev_act_id==actions_inv['RotateRight']:
                    act_id = LEFT
            elif self.acts is None:
                act_id = None
            else:
                act_id = next(self.acts, None)
            if act_id is None:
                act_id = DONE
                # num_times = 0
                while self.execution.qsize() > 0:
                    op = self.execution.get()
                    self.acts = op()
                    if self.acts is not None:
                        act_id = next(self.acts, None)
                        if act_id is not None:
                            break
            if act_id is None:
                act_id = DONE
        
        if False: #fig is not None:
            self._vis(fig, rgb, depth, act_id, point_goal)

        if type(act_id) != int:
            if act_id[0] in actions_inv:
                act_id = actions_inv[act_id[0]]
                self.prev_act_id = act_id
            else:
                self.prev_act_id = None
                st()
                return act_id

        self.prev_act_id = act_id

        action, param = self._get_action(act_id)

        return action, param

    def get_clostest_reachable_map_pos(self, map_pos):
        reachable = self._get_reachable_area()
        inds_i, inds_j = np.where(reachable)
        reachable_where = np.stack([inds_i, inds_j], axis=0)
        dist = distance.cdist(np.expand_dims(map_pos, axis=0), reachable_where.T)
        argmin = np.argmin(dist)
        ind_i, ind_j = inds_i[argmin], inds_j[argmin]
        return ind_i, ind_j

    def get_mapper_occ(self, obs, global_downscaling):
        # head_tilt = obs.head_tilt
        depth = np.array(obs.depth_map_list[-1])
        counts2, is_valids2, inds2 = self.mapper.get_occupancy_vars(self.position, 
                                    self.rotation, 
                                    -self.head_tilt, 
                                    depth, global_downscaling)
        return counts2, is_valids2, inds2

    def _vis(self, fig, rgb, depth, act_id, point_goal):
        ax = []
        spec = gridspec.GridSpec(ncols=2, nrows=2, 
            figure=fig, left=0., right=1., wspace=0.05, hspace=0.5)
        ax.append(fig.add_subplot(spec[0, 0]))
        ax.append(fig.add_subplot(spec[0, 1]))
        ax.append(fig.add_subplot(spec[1, 1]))
        dd = '\n'.join(wrap(self.goal.description, 50))
        fig.suptitle(f"{self.step_count-1}. {dd} act: {act_id}", fontsize=14)
        
        for a in ax:
            a.axis('off')
        
        m_vis = np.invert(self.mapper.get_traversible_map(
                          self.selem, 1,loc_on_map_traversible=True))
        explored_vis = self.mapper.get_explored_map(self.selem, 1)
        ax[0].imshow(rgb)
        ax[1].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                     cmap='Reds')
        state_xy = self.mapper.get_position_on_map()
        state_theta = self.mapper.get_rotation_on_map()
        arrow_len = 2.0/self.mapper.resolution
        ax[1].arrow(state_xy[0], state_xy[1], 
                    arrow_len*np.cos(state_theta+np.pi/2),
                    arrow_len*np.sin(state_theta+np.pi/2), 
                    color='b', head_width=20)
        if self.point_goal is not None:
            ax[1].plot(self.point_goal[0], self.point_goal[1], color='blue', marker='o',linewidth=10, markersize=12)
        #ax[2].set_title(f"Traversable {self.unexplored_area}")
        ax[1].set_title("Obstacle Map")
        ax[2].imshow(explored_vis > 0, origin='lower')
        ax[2].set_title('Explored Area')
        fig.savefig(f'images/{self.step_count}')

    def _cover_fn(self, uuid, iters, semantic_size_threshold, morph_disk_size):
        self.exploring = True
        unexplored = self._get_unexplored()
        if iters == 0:
            logging.error(f'Coverage iteration limit reached.')
            self.exploring = False

            return
        else:
            print("Unexplored", np.sum(unexplored))
            explored = np.sum(unexplored) < self.explored_threshold
            if explored:
                self.exploring = False
                logging.error(f'Unexplored area < {self.explored_threshold}. Exploration finished')
                if do_video:
                    cv2.destroyAllWindows()
                    self.video_writer.release()
                    self.video_ind += 1
            else:
                ind_i, ind_j = self._sample_point_in_unexplored_reachable(unexplored)
                self.point_goal = [ind_i, ind_j]
                
                print(f'Exploration setting pointgoal: {ind_i}, {ind_j}')
                fn = lambda: self._cover_fn(uuid, iters-1, semantic_size_threshold, morph_disk_size)
                self.execution.put(fn)
                fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), explore_mode=True, iters=self.max_steps_pointnav_cover)
                self.execution.put(fn)

    def _rotate_look_down_up(self, uuid):
        '''
        Function to rotate toward a box center and look down and then up
        '''
        object_on_map = self.mapper.get_object_on_map(uuid)
        disk = skimage.morphology.disk(1)
        object_on_map = skimage.morphology.binary_opening(object_on_map, disk)
        if object_on_map.sum() == 0:
            return
        if object_on_map.sum() == 400*600:
            object_on_map = object_on_map
        else:
            object_on_map = self._get_largest_cc(object_on_map)
        y, x = np.where(object_on_map)
        obj_x = np.mean(x)
        obj_y = np.mean(y)
        agent_x, agent_y = self.mapper.get_position_on_map()
        agent_theta = np.rad2deg(self.mapper.get_rotation_on_map())
        angle = np.rad2deg(np.arctan2(obj_y-agent_y, obj_x-agent_x))
        delta_angle = (angle-90-agent_theta) % 360
        if delta_angle <= 180:
            for _ in range(int(delta_angle)//self.DT):
                yield 'RotateLeft', {'rotation': self.DT}
        else:
            for _ in range(int((360 - delta_angle)//self.DT)):
                yield 'RotateRight', {'rotation': self.DT}
        max_forward = 5
        forwards = 0
        while self.obs.return_status == "SUCCESSFUL" and forwards < max_forward:
            forwards += 1
            yield self._get_action(FORWARD)
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        for _ in range(3):
            yield self._get_action(BACKWARD)

    def _check_point_goal_reached(self, goal_loc_cell, dist_thresh=0.5):
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        dist = np.sqrt(np.sum(np.square(state_xy - goal_loc_cell)))
        reached = dist*self.mapper.resolution < dist_thresh
        return reached

    def get_path_to_goal(self):
        '''
        Must call set_point_goal first
        '''

        traversible = self.mapper.get_traversible_map(self.selem, POINT_COUNT, loc_on_map_traversible=True)
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        if self.use_FMM_planner:
            state_theta = self.mapper.get_rotation_on_map() + np.pi/2
        else:
            state_theta = self.mapper.get_rotation_on_map()
        if self.use_FMM_planner:
            planner = FMMPlanner(traversible, 360//self.DT, int(self.STEP_SIZE/self.mapper.resolution), self.obstructed_actions)
        else:
            planner = ShortestPathPlanner(traversible, self.DT, self.STEP_SIZE, self.mapper.resolution, self.actions_inv, np.asarray(self.obstructed_states), np.stack(np.where(self.mapper.loc_on_map),1), step_count=self.step_count)
        ind_i, ind_j = self.point_goal
        goal_loc_cell = np.array([ind_j, ind_i])
        goal_loc_cell = goal_loc_cell.astype(np.int32)
        reachable = planner.set_goal(goal_loc_cell)

        a, state, act_seq, path = planner.get_action(np.array([state_xy[0], state_xy[1], state_theta]), self.steps_since_previous_failure>=5)
        return act_seq, path

    def _point_goal_fn_assigned(self, goal_loc_cell, explore_mode=False, dist_thresh=0.5, iters=100, search_mode=False, search_mode_reduced=False):
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        if self.use_FMM_planner:
            state_theta = self.mapper.get_rotation_on_map() + np.pi/2
        else:
            state_theta = self.mapper.get_rotation_on_map()

        dist = np.sqrt(np.sum(np.square(np.squeeze(state_xy) - np.squeeze(goal_loc_cell))))
        reached = self._check_point_goal_reached(goal_loc_cell, dist_thresh)
        if reached: 
            print("REACHED")
            if explore_mode:
                num_rot = int(np.floor((60 - self.head_tilt)/self.HORIZON_DT))
                for _ in range(num_rot):
                    yield actions[DOWN], {'horizon': self.HORIZON_DT}
                for _ in range(360//self.DT):
                    yield actions[LEFT], {'rotation': self.DT}
                for _ in range(num_rot):
                    yield actions[UP], {'horizon': self.HORIZON_DT}
            return 
        else:
            if iters==0:
                return
            traversible = self.mapper.get_traversible_map(self.selem, POINT_COUNT, loc_on_map_traversible=True)

            if self.use_FMM_planner:
                planner = FMMPlanner(traversible, 360//self.DT, int(self.STEP_SIZE/self.mapper.resolution), self.obstructed_actions)
            else:
                planner = ShortestPathPlanner(traversible, self.DT, self.STEP_SIZE, self.mapper.resolution, self.actions_inv, np.asarray(self.obstructed_states), np.stack(np.where(self.mapper.loc_on_map),1), step_count=self.step_count)

            goal_loc_cell = goal_loc_cell.astype(np.int32)
            reachable = planner.set_goal(goal_loc_cell)
            
            if self.use_FMM_planner:
                self.fmm_dist = planner.fmm_dist*1.
            if reachable[state_xy[1], state_xy[0]]:
                a, state, act_seq, path = planner.get_action(np.array([state_xy[0], state_xy[1], state_theta]), self.steps_since_previous_failure>=5)
                self.act_seq = act_seq
                if len(act_seq)==0 or act_seq[0] == 0:
                    print("Completed action sequence!")
                    if explore_mode:
                        num_rot = int(np.floor((60 - self.head_tilt)/self.HORIZON_DT))
                        for _ in range(num_rot):
                            yield actions[DOWN], {'horizon': self.HORIZON_DT}
                        for _ in range(360//self.DT):
                            yield actions[LEFT], {'rotation': self.DT}
                        for _ in range(num_rot):
                            yield actions[UP], {'horizon': self.HORIZON_DT}
                    return
                else:
                    pass

                # Fast Rotation (can't do fast rotation with fixed step size)
                if False:
                    rotations=act_seq[:-1]
                    if len(rotations)>0:
                        ty=rotations[0]
                        assert all(map(lambda x: x==ty,rotations)), 'bad acts'
                        ang = self.params[ty]['rotation'] * len(rotations)
                        yield actions[ty], {'rotation': ang}
                    yield FORWARD
                else:
                    for a in act_seq[0:1]:
                        yield a
                        if search_mode or (self.exploring and self.search_pitch_explore):
                            yield actions[DOWN], {'horizon': self.HORIZON_DT}
                            yield actions[DOWN], {'horizon': self.HORIZON_DT}
                            yield actions[UP], {'horizon': self.HORIZON_DT}
                            yield actions[UP], {'horizon': self.HORIZON_DT}
                        elif search_mode_reduced:
                            yield actions[DOWN], {'horizon': self.HORIZON_DT}
                            yield actions[UP], {'horizon': self.HORIZON_DT}

                fn = lambda: self._point_goal_fn_assigned(goal_loc_cell, search_mode=search_mode, explore_mode=False, dist_thresh=dist_thresh, iters=iters-1)
                self.execution.put(fn)

    def _get_reachable_area(self):
        traversible = self.mapper.get_traversible_map(self.selem, POINT_COUNT, loc_on_map_traversible=True)
        traversible = skimage.morphology.binary_opening(traversible, self.selem)
        if self.use_FMM_planner:
            planner = FMMPlanner(traversible, 360//self.DT, int(self.STEP_SIZE/self.mapper.resolution), self.obstructed_actions)
        else:
            planner = ShortestPathPlanner(traversible, self.DT, self.STEP_SIZE, self.mapper.resolution, self.actions_inv, np.asarray(self.obstructed_states), np.stack(np.where(self.mapper.loc_on_map),1), step_count=self.step_count)
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        reachable = planner.set_goal(state_xy)
        return reachable

    def _get_unexplored(self):
        reachable = self._get_reachable_area()
        explored_point_count = 1
        explored = self.mapper.get_explored_map(self.selem, explored_point_count)
        unexplored = np.invert(explored)
        unexplored = np.logical_and(unexplored, reachable)
        # added to remove noise effects
        disk = skimage.morphology.disk(2)
        unexplored = skimage.morphology.binary_opening(unexplored, disk)
        self.unexplored_area = np.sum(unexplored)
        return unexplored

    def _sample_point_in_unexplored_reachable(self, unexplored):
        ind_i, ind_j = np.where(unexplored)
        ind = self.rng.randint(ind_i.shape[0])
        return ind_i[ind], ind_j[ind]
