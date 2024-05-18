import numpy as np
import os
from arguments import args
import random
import math
from tensorboardX import SummaryWriter
from ai2thor_docker.ai2thor_docker.x_server import startx
import torch
import utils.aithor
import utils.geom
from PIL import Image
import ipdb
st = ipdb.set_trace
import matplotlib.pyplot as plt
from task_base.aithor_base import Base
from itertools import cycle
# from task_base.messup import mess_up, mess_up_from_loaded

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class Ai2Thor_Base(Base):

    def __init__(self):   

        super(Ai2Thor_Base, self).__init__()

    def run_agent(self, load_dict=None):
        outputs = self.run_agent_solq(load_dict=load_dict)
        return outputs

    def run_agent_solq(self, load_dict=None, z=[0.05, 2.0]):
        '''
        max_openable_per_cat: how many max of each category to get observations for
        '''
        
        if args.randomize_object_placements:
            print("Randomizing object placement")
            # randomize scene
            random_seed = np.random.randint(1000)
            self.controller.step(action="InitialRandomSpawn",
                randomSeed=random_seed,
                forceVisible=True,
                numPlacementAttempts=5,
                placeStationary=False,
            )

        # if args.closed_prob is not None:
        if args.randomize_object_state:
            print("Randomizing object states")
            # print("opening some objects...")
            for obj_meta in self.controller.last_event.metadata['objects']: #objects:

                # obj_category_name = obj_meta['objectType']
                # print(obj_category_name)

                # if obj_category_name not in self.name_to_id:
                #     continue

                if obj_meta['openable']:
                    openness_increments = 0.1
                    openness_degrees = [-1] + list(np.round(np.arange(0,1+openness_increments,openness_increments),decimals=1))
                    increments = openness_degrees[1:]
                    closed_prob = 0.5
                    p_not_closed = (1-closed_prob)/len(increments[1:])
                    p = list(np.ones(len(increments[1:]))*p_not_closed) # keep probability of closed fixed (a bit higher than the rest)
                    p = [closed_prob] + p 
                    openness = np.random.choice(increments, p=p)
                    openness = np.round(openness, decimals=1)
                    # print("opening", obj_meta['objectId'], "by", openness)
                    self.controller.step(
                        action="OpenObject",
                        objectId=obj_meta['objectId'],
                        openness=openness,
                        forceAction=True # set to True otherwise agent needs to be close to it
                    )

                if obj_meta['cookable']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.8:
                        self.controller.step(
                            action="CookObject",
                            objectId=obj_meta['objectId'],
                            forceAction=True
                        )

                if obj_meta['canFillWithLiquid']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.8:
                        liquid = np.random.choice(np.array(["coffee", "wine", "water"]))
                        self.controller.step(
                            action="FillObjectWithLiquid",
                            objectId=obj_meta['objectId'],
                            fillLiquid=liquid,
                            forceAction=True
                        )

                if obj_meta['sliceable']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.7:
                        self.controller.step(
                            action="SliceObject",
                            objectId=obj_meta['objectId'],
                            forceAction=True
                        )

                if obj_meta['toggleable']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.8:
                        self.controller.step(
                            action="ToggleObjectOn",
                            objectId=obj_meta['objectId'],
                            forceAction=True
                        )

                if obj_meta['dirtyable']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.8:
                        self.controller.step(
                            action="DirtyObject",
                            objectId=obj_meta['objectId'],
                            forceAction=True
                        )


        # if args.do_predict_oop:
        #     if args.mess_up_from_loaded:
        #         object_dict = load_dict['object_dict'] 
        #         oop_IDs = load_dict['oop_IDs'] 
        #         object_messup_meta = load_dict['objects_messup'] 
        #         mess_up_from_loaded(self.controller, object_messup_meta)
        #         print("MESSED UP FROM LOADED")
        #     else:
        #         object_dict, oop_IDs = mess_up(self.controller, self.include_classes, args.num_objects)
        # else:
        object_dict = None     

                        
        rgb_scene_batch, targets, multiview_batch = [], [], []          

        objects_meta_all = self.controller.last_event.metadata['objects']

        random.shuffle(objects_meta_all)

        # st()
        objects_meta_all_cycle = cycle(objects_meta_all)
        
        # valid_pts_selected = []
        # loop through objects
        successes = 0
        attempts = 0
        object_names = []
        view_ids = []
        while True:

            if args.randomize_scene_lighting_and_material:
                print("Randomizing materials")
                
                # prob = np.random.uniform(0, 1)
                # if prob>0.4:
                #     # some augmentations
                #     self.controller.step(
                #         action="RandomizeMaterials",
                #         useTrainMaterials=None,
                #         useValMaterials=None,
                #         useTestMaterials=None,
                #         inRoomTypes=None
                #     )
                
                prob = np.random.uniform(0, 1)
                if prob>0.5:
                    self.controller.step(
                        action="RandomizeLighting",
                        brightness=(0.5, 1.5),
                        randomizeColor=True,
                        hue=(0, 1),
                        saturation=(0.5, 1),
                        synchronized=False
                    )

            observations = {}

            # if args.do_predict_oop:
            #     # select random object to target in view
            #     oop_ind = np.random.randint(len(oop_IDs))
            #     oop_ID_select = oop_IDs[oop_ind]
            #     obj = object_dict[oop_ID_select]['meta']
            # else:
            obj = next(objects_meta_all_cycle)

            print("BATCH SIZE", args.data_batch_size)
            
            break_crit = successes >= args.data_batch_size
            if break_crit:
                break

            # here we want spawn agent near an object
            obj_category_name = obj['objectType']
            # print(obj_category_name)

            if obj_category_name not in self.name_to_id:
                continue

            print("obtaining batch ", successes) 

            # Calculate distance to object center
            obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))
            # aithor_pos_camX = obj_center

            self.controller.step(action="GetReachablePositions")
            reachable = self.controller.last_event.metadata["actionReturn"]
            reachable = np.array([list(d.values()) for d in reachable])
            distances = np.sqrt(np.sum((reachable - np.expand_dims(obj_center, axis=0))**2, axis=1))
            map_pos = obj_center

            within_radius = np.logical_and(distances > args.radius_min, distances < args.radius_max)
            valid_pts = reachable[within_radius]

            found_one = self.find_starting_viewpoint(valid_pts, obj_center)
            if not found_one:
                continue

            movement_util = Movement_Util()

            targets_ = []
            rgb_scene_batch_ = []
            origin_T_camX_batch_ = []
            depth_batch_ = []
            # multiview_ = []
            fail = True
            views_obtained = 0
            for s in range(args.S+args.views_to_attempt):
                '''
                generate trajectory by taking random actions
                '''
                
                movement_util.move_agent(self.controller, mode=args.movement_mode)
                                
                obs = self.get_observations(object_dict=object_dict, fail_if_no_objects=args.fail_if_no_objects)

                if obs is None:
                    movement_util.movement_failed = True
                    movement_util.undo_previous_movement(self.controller)
                    continue # try again

                targets_.append(obs["target_frame"])
                rgb_scene_batch_.append(obs["rgb_torch"])

                views_obtained += 1

                if views_obtained==args.S:
                    fail = False
                    break

            if fail:
                print("failed. let's try again..")
                continue   
            if len(targets_)==0:
                print("failed. let's try again..")
                continue

            targets.extend(targets_)
            rgb_scene_batch.extend(rgb_scene_batch_)

            successes += 1
            attempts = 0

        if len(rgb_scene_batch)==0:
            return None

        rgb_scene_batch = torch.stack(rgb_scene_batch).float()
        
        outputs = [rgb_scene_batch, targets]

        return outputs 

    def find_starting_viewpoint(self, valid_pts, obj_center):
        '''
        Finds a "good" starting viewpoint based on object center in Aithor 3D coordinates and valid spawn points
        '''
        found_one = False
        if valid_pts.shape[0]==0:
            return found_one
        for i in range(10): # try 10 times to find a good location
            s_ind = np.random.randint(valid_pts.shape[0])
            pos_s = valid_pts[s_ind]

            # add height from center of agent to camera
            pos_s[1] = pos_s[1] + 0.675

            # YAW calculation - rotate to object
            agent_to_obj = np.squeeze(obj_center) - pos_s 
            agent_local_forward = np.array([0, 0, 1.0]) 
            flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
            flat_dist_to_obj = np.linalg.norm(flat_to_obj) + 1e-6
            flat_to_obj /= flat_dist_to_obj

            det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
            turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

            # add noise so not right in the center
            noise = np.array([0.0, 0.0]) #np.random.normal(0, 2, size=2)

            turn_yaw = np.degrees(turn_angle) + noise[0]
            turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj)) + noise[1]

            turn_yaw = np.round(turn_yaw/args.DT) * args.DT

            # just add a little noise to the pitch
            rand_pick = np.random.uniform(0,1)
            if rand_pick < 0.1:
                turn_pitch = np.floor(turn_pitch/args.HORIZON_DT) * args.HORIZON_DT
            elif rand_pick > 0.9:
                turn_pitch = np.ceil(turn_pitch/args.HORIZON_DT) * args.HORIZON_DT
            else:
                turn_pitch = np.round(turn_pitch/args.HORIZON_DT) * args.HORIZON_DT

            # move agent and rotate to object
            event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1] - 0.675, z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch), standing=True)

            # print("TURN PITCH:", turn_pitch)

            found_one = True
            break

        return found_one

    def get_observations(self, object_dict=None, verbose=False, fail_if_no_objects=True, shuffle_inds=True):
        rgb = self.controller.last_event.frame
        # depth = event.depth_frame

        # plt.figure(1); plt.clf()
        # plt.imshow(rgb)
        # plt.savefig('images/test.png')
        # st()

        rgb_PIL = Image.fromarray(rgb)
        rgb_norm = rgb.astype(np.float32) * 1./255
        rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1)#.cuda() #to(self.device0) #.cuda()

        origin_T_camX = utils.aithor.get_origin_T_camX(self.controller.last_event, False)#.cuda()

        semantic = self.controller.last_event.instance_segmentation_frame
        object_id_to_color = self.controller.last_event.object_id_to_color
        color_to_object_id = self.controller.last_event.color_to_object_id
        
        obj_metadata_IDs = []
        for obj_m in self.controller.last_event.metadata['objects']: #objects:
            obj_metadata_IDs.append(obj_m['objectId'])

        obj_boxes = []
        obj_catnames = []
        object_ids = []
        object_ids2 = []
        object_ids_openness = []
        obj_segs = []
        obj_names = []

        instance_masks = self.controller.last_event.instance_masks
        instance_detections2d = self.controller.last_event.instance_detections2D
        obj_meta_all = self.controller.last_event.metadata['objects']
        masks = []
        bboxes = []
        labels = []
        centroids = []
        filter_inds = []
        visibilities = []

        idxs = []
        for object_id in instance_masks.keys(): #range(obj_ids.shape[0]): 

            if object_id not in obj_metadata_IDs:
                continue

            idxs.append(object_id)  

        if shuffle_inds:
            random.shuffle(idxs)

        for object_id in idxs: 

            obj_meta_index = obj_metadata_IDs.index(object_id)
            obj_meta = obj_meta_all[obj_meta_index]

            obj_category_name = obj_meta['objectType']

            if obj_category_name not in self.name_to_id:
                continue

            obj_name = obj_meta['name']

            i_mask = instance_masks[object_id]
            num_points = np.sum(i_mask)

            obj_bbox = instance_detections2d[object_id]

            # remove objects that are very small
            if num_points < (args.min_percent_points*self.W*self.H):
                continue
            
            obj_3dbox_origin = utils.aithor.get_3dbox_in_geom_format(obj_meta)
            # get amodal box
            boxlist2d_amodal, obj_3dbox_camX = utils.aithor.get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(self.pix_T_camX).unsqueeze(0).cuda(), self.H, self.W)
            boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
            
            boxlist2d_amodal_clip = np.zeros(4)
            boxlist2d_amodal_clip[[0,2]] = np.clip(boxlist2d_amodal[[0,2]], 0, self.W)
            boxlist2d_amodal_clip[[1,3]] = np.clip(boxlist2d_amodal[[1,3]], 0, self.H)
            iou_inview = float(np.squeeze(utils.box.boxlist_2d_iou(boxlist2d_amodal.reshape(1,4), boxlist2d_amodal_clip.reshape(1,4))))

            if False: # plot boxes
                # clist_orgin = torch.mean(obj_3dbox_origin, axis=2)
                # clist_camX = utils.geom.apply_4x4(camX_T_origin, clist_orgin)
                # center_pix = list(utils.geom.apply_pix_T_cam(torch.from_numpy(self.pix_T_camX).unsqueeze(0), clist_camX).squeeze().cpu().numpy().astype(np.int32))
                # center_pix[1] = 256-center_pix[1]
                # center_pix = tuple(center_pix)

                # img = rgb.copy()
                # cv2.circle(img,center_pix, 2, (0,0,255), -1)
                # plt.imshow(img)
                # plt.savefig('images/test3.png')

                img = rgb.copy()
                cv2.rectangle(img, (int(boxlist2d_amodal[0]), int(boxlist2d_amodal[1])), (int(boxlist2d_amodal[2]), int(boxlist2d_amodal[3])),color=(0, 255, 0), thickness=1)
                cv2.rectangle(img, (int(obj_bbox[0]), int(obj_bbox[1])), (int(obj_bbox[2]), int(obj_bbox[3])),color=(255, 0, 0), thickness=1)
                plt.imshow(img)
                plt.savefig('images/test.png')

                rgb_box3d = self.summ_writer.summ_box_by_corners_aithor('test', rgb_torch.cpu().unsqueeze(0) - 0.5, obj_3dbox_camX.cpu(), torch.ones((1,1)), torch.ones((1,1)), torch.from_numpy(self.pix_T_camX).unsqueeze(0), only_return=True)
                rgb_box3d = (rgb_box3d + 0.5) *255. 
                rgb_box3d = rgb_box3d.cpu().squeeze(0).permute(1,2,0).numpy().astype(np.uint)
                plt.imshow(rgb_box3d)
                plt.savefig('images/test2.png')
                print("HERE")
                st()

            area_amodal = (boxlist2d_amodal_clip[2] - boxlist2d_amodal_clip[0]) * (boxlist2d_amodal_clip[3] - boxlist2d_amodal_clip[1])
            area_modal = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
            percent_occluded = area_modal/area_amodal

            visibility = min(iou_inview, percent_occluded) # percent of object in frame, percent occluded

            # remove anything not visible enough
            if args.visibility_threshold is not None:
                if visibility < args.visibility_threshold:
                    continue

            if args.amodal:
                obj_bbox = boxlist2d_amodal_clip
            else:
                obj_bbox = obj_bbox # [start_x, start_y, end_x, end_y]    

            center_x = ((obj_bbox[0] + obj_bbox[2]) / 2) / self.W
            center_y = ((obj_bbox[1] + obj_bbox[3]) / 2) / self.H
            width = (obj_bbox[2] - obj_bbox[0]) / self.W
            height = (obj_bbox[3] - obj_bbox[1]) / self.H
            obj_bbox_coco_format = torch.from_numpy(np.array([center_x, center_y, width, height]))

            if args.use_masks:
                obj_segs.append(torch.from_numpy(i_mask))
            obj_boxes.append(obj_bbox_coco_format)
            obj_catnames.append(obj_category_name)
            obj_names.append(obj_name)
            # visibilities.append(torch.tensor([visibility]))

            # if args.do_predict_oop:
            #     object_ids.append(torch.tensor([self.name_to_id[obj_category_name]]))
            #     object_ids2.append(torch.tensor([self.label_to_id[object_dict[obj_meta['name']]['out_of_place']]]))
            # else:
            object_ids.append(torch.tensor([self.name_to_id[obj_category_name]]))
            
        
        print("Number of objects found:", len(object_ids))
        if fail_if_no_objects:
            if len(object_ids)==0: # want atleast one object
                if verbose:
                    print("No objects found")
                # fail = True
                return None
                # print("not enough objects")
                # continue
        if len(object_ids)>0:
            obj_boxes = torch.stack(obj_boxes).float() #.cuda()
            object_ids = torch.cat(object_ids, dim=0).long() #.cuda()
            # visibilities = torch.cat(visibilities, dim=0).float() #.cuda()
            obj_segs = torch.stack(obj_segs)
        else:
            return None
        
        target_frame = {}
        target_frame['boxes'] = obj_boxes
        # target_frame['visibility'] = visibilities
        if args.use_masks:
            obj_masks = obj_segs #.cuda()
            target_frame['masks'] = obj_masks
        
        obs = {}
        # if args.do_predict_oop:
        #     # for this we want first head to predict semantic
        #     object_ids2 = torch.cat(object_ids2, dim=0)#.cuda()
        #     target_frame['labels'] = object_ids.long() # semantics
        #     target_frame['labels2'] = object_ids2.long() # oop 
        #     # print("num objects:", len(object_ids2))
        #     # print("num oop:", sum(object_ids2))
        # else:
        target_frame['labels'] = object_ids
            
        obs["rgb_torch"] = rgb_torch
        obs["target_frame"] = target_frame

        return obs


class Movement_Util():

    def __init__(self):   

        self.previous_movement = None
        self.movement_failed = False
        self.rotate_inc = 0.0
        self.move_inc = 0.0

    def undo_previous_movement(self, controller):
        if self.previous_movement is not None:
            print("Undoing movement")
            movements_undo = {"MoveAhead":"MoveBack", "RotateRight":"RotateLeft", "RotateLeft":"RotateRight", "LookUp":"LookDown", "LookDown":"LookUp"}
            movement = movements_undo[self.previous_movement]
            if "Rotate" in self.previous_movement:
                controller.step(
                    action=movement,
                    degrees=args.DT + self.rotate_inc
                )
            elif "Look" in self.previous_movement:
                controller.step(
                    action=movement,
                    degrees=args.HORIZON_DT + self.rotate_inc
                )
            else:
                controller.step(
                    action=movement,
                    moveMagnitude=args.STEP_SIZE + self.move_inc
                )

    def move_agent(self, controller, mode="forward_first"):

        if mode=="forward_first":
            movement = "MoveAhead"
            turn_right_or_left = np.random.randint(4)
            if turn_right_or_left==0:
                turn_dir = "RotateRight"
            elif turn_right_or_left==1:
                turn_dir = "RotateLeft"
            elif turn_right_or_left==2:
                turn_dir = "LookDown"
            else:
                turn_dir = "LookUp"
                
            position = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
            rotation = np.array(list(controller.last_event.metadata["agent"]["rotation"].values()))

            # always try moving forward
            controller.step(
                action=movement,
                moveMagnitude=args.STEP_SIZE
            )
            
            position_ = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
            rotation_ = np.array(list(controller.last_event.metadata["agent"]["rotation"].values()))

            # didn't move forward, then rotate
            if np.sum(position - position_)==0:
                if "Look" in turn_dir:
                    controller.step(
                        action=turn_dir,
                        degrees=args.HORIZON_DT
                    )
                else:
                    controller.step(
                        action=turn_dir,
                        degrees=args.DT
                    )
                self.previous_movement = turn_dir
            else:
                self.previous_movement = movement
            
            position_ = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
            rotation_ = np.array(list(controller.last_event.metadata["agent"]["rotation"].values()))

            pos_diff = position_ - position 
            rot_diff = rotation_ - rotation

            # if sum(pos_diff) == 0 and sum(rot_diff) == 0:
            #     continue
            # print("pos_diff", pos_diff)
            # print("rot_diff", rot_diff)
        elif mode=="random":
            movements = ["MoveAhead", "RotateRight", "RotateLeft"]
            movement_idxs = np.arange(3)
            np.random.shuffle(movement_idxs)
            for movement_idx in list(movement_idxs):
                movement = movements[movement_idx]
                if "Rotate" in movement:
                    controller.step(
                        action=movement,
                        degrees=args.DT
                    )
                else:
                    controller.step(
                        action=movement,
                        moveMagnitude=args.STEP_SIZE
                    )
                if controller.last_event.metadata['lastActionSuccess']:
                    break
        elif mode=="random_with_momentum":
            
            p = np.random.uniform(0,1)
            if p > 0.5 and (self.previous_movement is not None) and (not self.movement_failed):
                movement = self.previous_movement
                if "Rotate" in movement:
                    rotate_inc_ = np.abs(np.random.normal(0, args.DT/4))
                    self.rotate_inc += rotate_inc_
                    controller.step(
                        action=movement,
                        degrees=args.DT + self.rotate_inc,
                    )
                elif "Look" in movement:
                    rotate_inc_ = np.abs(np.random.normal(0, args.HORIZON_DT/4))
                    self.rotate_inc += rotate_inc_
                    controller.step(
                        action=movement,
                        degrees=args.HORIZON_DT + self.rotate_inc,
                    )
                else:
                    move_inc_ = np.abs(np.random.normal(0, args.STEP_SIZE/4))
                    self.move_inc += move_inc_
                    controller.step(
                        action=movement,
                        moveMagnitude=args.STEP_SIZE + self.move_inc,
                    )

            else:
                movements = ["MoveAhead", "RotateRight", "RotateLeft", "LookUp", "LookDown"]
                movement_idxs = np.arange(len(movements))
                np.random.shuffle(movement_idxs)
                for movement_idx in list(movement_idxs):
                    movement = movements[movement_idx]
                    if "Rotate" in movement:
                        self.rotate_inc = np.abs(np.random.normal(0, args.DT/4))
                        controller.step(
                            action=movement,
                            degrees=args.DT + self.rotate_inc
                        )
                    elif "Look" in movement:
                        self.rotate_inc = np.abs(np.random.normal(0, args.HORIZON_DT/4))
                        controller.step(
                            action=movement,
                            degrees=args.HORIZON_DT + self.rotate_inc
                        )
                    else:
                        self.move_inc = np.abs(np.random.normal(0, args.STEP_SIZE/4))
                        controller.step(
                            action=movement,
                            moveMagnitude=args.STEP_SIZE + self.move_inc
                        )
                    if controller.last_event.metadata['lastActionSuccess']:
                        break
                self.previous_movement = movement
            self.movement_failed = False



        else:
            assert(False) # wrong movement mode

        event = controller.last_event
        return event