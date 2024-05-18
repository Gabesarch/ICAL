from utils.aithor import compute_metrics, read_task_data
import numpy as np
from arguments import args
import skimage.morphology
import ipdb
st = ipdb.set_trace
if args.mode in ["teach_eval_tfd", "teach_eval_custom", "teach_eval_continual"]:
    from teach.inference.tfd_inference_runner import TfdInferenceRunner as InferenceRunner
elif args.mode=="teach_eval_edh":
    from teach.inference.edh_inference_runner import EdhInferenceRunner as InferenceRunner
else:
    from teach.inference.tfd_inference_runner import TfdInferenceRunner as InferenceRunner
import matplotlib.pyplot as plt

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

class TeachTask():
    def __init__(
        self, 
        env, 
        action_to_mappedaction=None,
        object_instance_ids=None,
        max_steps=1000,
        max_fails=30,
        object_tracker=None,
        approx_last_action_success=False,
        use_GT_error_feedback=False,
        remove_unusable_slice=False,
        ):
        self.env = env
        self.steps = 0
        self.action_to_mappedaction = action_to_mappedaction
        self.num_fails = 0
        self.max_steps = max_steps
        self.max_fails = max_fails
        self.remove_unusable_slice = remove_unusable_slice
        self.approx_last_action_success = approx_last_action_success
        self.use_GT_error_feedback = use_GT_error_feedback
        self.last_action = None
        self.object_tracker = object_tracker
        if self.approx_last_action_success:
            self.success_detector = CheckSuccessfulAction()
        # if not self.use_GT_error_feedback:
        #     self.error_feedback = ErrorFeedback()
        self.pointer = None
        self.metrics = None
        self.interacted_objects = set()

        self.reward = 0
        self.args = args
        self.err_message = None
        self.help_message = None

        self.force_actions=args.force_actions

        self.interaction_images = []
    
    def get_observations(self):
        rgb = self.env.simulator.controller.last_event.frame
        obs = {}
        obs["rgb"] = rgb 
        if not args.use_estimated_depth:
            depth = self.env.simulator.controller.last_event.depth_frame
            obs["depth"] = depth #* (1/1000) # convert to meters
        obs["is_holding"] = True # assume always holding - sometimes agent's head is in view, and we need to filter this out of depth frame
        return obs

    def get_observation_history(self):
        return self.obs_history

    def action_success(self):
        if self.approx_last_action_success:
            success = self.success_detector.check_successful_action(self.env.simulator.controller.last_event.frame, self.last_action)
        else:
            success = self.sim_succ
        return success

    def get_agent_head_tilt(self):
        return self.env.simulator.controller.last_event.metadata["agent"]["cameraHorizon"]

    def is_done(self):
        return True if (self.steps>=self.max_steps or self.num_fails>=self.max_fails) else False

    def step(self, action, obj_relative_coord=None, object_name=None, log_error=True):

        if self.action_to_mappedaction is not None and action in self.action_to_mappedaction.keys():
            action = self.action_to_mappedaction[action]

        cmd = {}
        cmd["action"] = action
        self.last_action = action
        if self.approx_last_action_success:
            self.success_detector.update_image(self.env.simulator.controller.last_event.frame)

        if self.is_done():
            print(f"Max steps ({self.steps}; max={self.max_steps}) or max fails ({self.num_fails}; max={self.max_fails}) reached! Returning without interaction..")
            return

        if args.simulate_actions:
            print("Simulating (Fake) action...")
            sim_succ, err_message, help_message = True, '', ''
        else:
            # print("Real action!")
            sim_succ, err_message, help_message = InferenceRunner._execute_action(self.env.simulator, action, obj_relative_coord, force=self.force_actions)
        self.sim_succ = sim_succ

        if not args.simulate_actions:

            if action in ["Place", "Pickup", 'Open', 'Close', "ToggleOn", "ToggleOff", "Slice", "Pour"]:
                if args.save_memory_images:
                    self.interaction_images.append(self.get_observations()["rgb"])
                if self.env.simulator.controller.last_event.metadata["actionReturn"] is not None:
                    for object_ in self.env.simulator.controller.last_event.metadata["objects"]:
                        if object_["objectId"]==self.env.simulator.controller.last_event.metadata["actionReturn"]:
                            self.interacted_objects.add(object_["name"])
                else:
                    for object_ in self.env.simulator.controller.last_event.metadata["objects"]:
                        if object_["objectType"]==object_name:
                            self.interacted_objects.add(object_["name"])
                print(self.interacted_objects)

            print(f"Steps reached: {self.steps}/{self.max_steps} (action: {action})")

            if self.remove_unusable_slice and sim_succ and action=="Slice":
                # plt.figure()
                # plt.imshow(self.env.simulator.controller.last_event.frame)
                # plt.savefig('output/test.png')
                largest_slice = None
                for obj in self.env.simulator.controller.last_event.metadata['objects']:
                    if "Slice" in obj["name"] and object_name in obj["name"]:
                        if largest_slice is None:
                            largest_slice = obj
                        elif np.prod(list(obj['axisAlignedBoundingBox']['size'].values()))>np.prod(list(largest_slice['axisAlignedBoundingBox']['size'].values())):
                            largest_slice = obj
                if largest_slice is not None:
                    self.env.simulator.controller.step(
                                action="RemoveFromScene",
                                objectId=largest_slice["objectId"]
                            )
                # plt.figure()
                # plt.imshow(self.env.simulator.controller.last_event.frame)
                # plt.savefig('output/test1.png')
                # st()
                for obj in self.env.simulator.controller.last_event.metadata['objects']:
                    if "Slice" in obj["name"] and object_name in obj["name"] and "Slice" not in obj["objectId"]:
                        # even though we sliced the object, some of the object remains unsliced and unusable? Let's remove this
                        self.env.simulator.controller.step(
                            action="RemoveFromScene",
                            objectId=obj["objectId"]
                        )

            if log_error:
                if self.use_GT_error_feedback:
                    self.err_message = err_message
                    self.help_message = help_message
                    if not self.sim_succ:
                        # if already toggled on, change error message
                        if action=="ToggleOn":
                            for obj in self.env.simulator.controller.last_event.metadata['objects']:
                                if obj['isToggled'] and obj['objectType']==object_name and obj['visible']:
                                    self.err_message = "Object is already toggled on."
                                    self.help_message = "You cannot toggle on an object that is already on."
                        elif action=="ToggleOff":
                            for obj in self.env.simulator.controller.last_event.metadata['objects']:
                                if (not obj['isToggled']) and obj['objectType']==object_name and obj['visible']:
                                    self.err_message = "Object is already toggled off."
                                    self.help_message = "You cannot toggle off an object that is already off."

                        if "not a valid Object Type to be placed" in err_message and "Bread" in err_message and "Toaster" in err_message:
                            self.err_message = "Bread is too big to fit into toaster. Cannot place it there."
                            self.help_message = "You should slice the bread thinner to place it in the toaster."
                else:
                    if not self.action_success():
                        if obj_relative_coord is None:
                            self.err_message = "The agent is blocked from moving in that direction."
                            self.help_message = "Find an alternate route or viewpoint."
                        else:
                            feedback = "" #self.error_feedback.get_error_message(self.env.simulator.controller.last_event.frame)
                            self.err_message = feedback
                            self.help_message = ""
                    else:
                        self.err_message = "The action was successful."
                        self.help_message = "Do nothing and carry on."
            
                if not sim_succ:
                    self.num_fails += 1
                    print(f"Number of failures: {self.num_fails}/{self.max_fails}")
                self.steps += 1

                self.event = self.env.simulator.controller.last_event

                if self.metrics is not None:
                    InferenceRunner._update_metrics(self.metrics, action, obj_relative_coord, sim_succ)

        return sim_succ

    def get_gt_masks(self, object_cat, name_to_mapped_name=None):
        
        obj_metadata_IDs = []
        for obj_m in self.env.last_event.metadata['objects']: #objects:
            obj_metadata_IDs.append(obj_m['objectId'])

        instance_masks = self.env.last_event.instance_masks
        obj_meta_all = self.env.last_event.metadata['objects']

        idxs = []
        for object_id in instance_masks.keys(): #range(obj_ids.shape[0]): 
            if object_id not in obj_metadata_IDs:
                continue
            idxs.append(object_id)  

        masks = []
        for object_id in idxs: 
            obj_meta_index = obj_metadata_IDs.index(object_id)
            obj_meta = obj_meta_all[obj_meta_index]
            obj_category_name = obj_meta['objectType']
            if name_to_mapped_name is not None and obj_category_name in name_to_mapped_name.keys():
                obj_category_name = name_to_mapped_name[obj_category_name]
            if obj_category_name!=object_cat:
                continue
            i_mask = instance_masks[object_id]
            masks.append(i_mask)
        if len(masks)==0:
            pass
        else:
            masks = np.stack(masks)

        return masks

class CheckSuccessfulAction():
    def __init__(self, rgb_init=None):
        '''
        rgb_init: the rgb image from the spawn viewpoint W, H, 3
        This class does a simple check with the previous image to see if it completed the action 
        '''
        self.rgb_prev = rgb_init

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

class ErrorFeedback():
    def __init__(self):
        '''
        CLIP error feedback
        '''
        from nets.clip import ALIGN
        self.model = ALIGN()
        with open('task_base/feedback.txt') as f:
            self.lines = [line.rstrip() for line in f]

    def get_error_message(self, rgb):
        probs = self.model.score(rgb, self.lines)
        argmax_error = np.argmax(probs.cpu().numpy())
        feedback = self.lines[argmax_error]
        return feedback



