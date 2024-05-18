# from map_and_plan.FILM.models.depth.alfred_perception_models import AlfredSegmentationAndDepthModel
import numpy as np
from collections import Counter, OrderedDict
from arguments import args
import torch
from torchvision import transforms
from PIL import Image
# from map_and_plan.FILM.models.segmentation.segmentation_helper import SemgnetationHelper
import cv2
import ipdb
st = ipdb.set_trace
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Depth():
    def __init__(self, estimate_depth=True, task=None, on_aws=False, DH=300, DW=300):
        self.task = task
        self.agent = self.task

        self.W = args.W
        self.H = args.H

        self.DH = DH
        self.DW = DW

        self.args = args

        self.estimate_depth = estimate_depth
        self.use_learned_depth = estimate_depth
        # self.use_sem_seg = args.use_sem_seg

        self.res = transforms.Compose([transforms.ToPILImage(),
					transforms.Resize((args.frame_height, args.frame_width),
									  interpolation = Image.NEAREST)])

        if self.estimate_depth:
            self.depth_gpu =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if not(args.valts_depth):
                bts_args = SimpleNamespace(model_name='bts_nyu_v2_pytorch_densenet161' ,
                                            encoder='densenet161_bts',
                                            dataset='alfred',
                                            input_height=300,
                                            input_width=300,
                                            max_depth=5,
                                            mode = 'test',
                                            device = self.depth_gpu,
                                            set_view_angle=False,
                                            load_encoder=False,
                                            load_decoder=False,
                                            bts_size=512)

                if self.args.depth_angle or self.args.cpp:
                    bts_args.set_view_angle = True
                    bts_args.load_encoder = True

                self.depth_pred_model = BtsModel(params=bts_args).to(device=self.depth_gpu)
                print("depth initialized")

                if args.cuda:
                    ckpt_path = '/projects/katefgroup/REPLAY/film_models/depth/depth_models/' + args.depth_checkpoint_path 
                else:
                    ckpt_path = '/projects/katefgroup/REPLAY/film_models/depth/depth_models/' + args.depth_checkpoint_path 
                checkpoint = torch.load(ckpt_path, map_location=self.depth_gpu)['model']

                new_checkpoint = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:] # remove `module.`
                    new_checkpoint[name] = v
                del checkpoint
                # load params
                self.depth_pred_model.load_state_dict(new_checkpoint)
                self.depth_pred_model.eval()
                self.depth_pred_model.to(device=self.depth_gpu)

            #Use Valts depth
            else:

                model_path ='valts/model-2000-best_silog_10.13741' #45 degrees only model

                if self.args.depth_model_old:
                    model_path ='valts/model-34000-best_silog_16.80614'

                elif self.args.depth_model_45_only:
                    model_path = 'valts/model-500-best_d3_0.98919'

                self.depth_pred_model = AlfredSegmentationAndDepthModel()

                state_dict = torch.load('/projects/katefgroup/REPLAY/film_models/depth/depth_models/' +model_path, map_location=self.depth_gpu)['model']

                new_checkpoint = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_checkpoint[name] = v

                state_dict = new_checkpoint
                del new_checkpoint

                self.depth_pred_model.load_state_dict(state_dict)
                self.depth_pred_model.eval()
                self.depth_pred_model.to(device=self.depth_gpu)

                if self.args.separate_depth_for_straight:
                    model_path = 'valts0/model-102500-best_silog_17.00430'

                    self.depth_pred_model_0 = AlfredSegmentationAndDepthModel()
                    state_dict = torch.load('/projects/katefgroup/REPLAY/film_models/depth/depth_models/' +model_path, map_location=self.depth_gpu)['model']

                    new_checkpoint = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_checkpoint[name] = v
                    
                    state_dict = new_checkpoint
                    del new_checkpoint

                    self.depth_pred_model_0.load_state_dict(state_dict)
                    self.depth_pred_model_0.eval()
                    self.depth_pred_model_0.to(device=self.depth_gpu)

            if self.use_sem_seg:
                self.seg = SemgnetationHelper(self.agent)

    def _preprocess_obs(self, obs):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]	

        # print(rgb.shape)

        if self.use_sem_seg:
            sem_seg_pred = self.seg.get_sem_pred(rgb.astype(np.uint8))
        else:
            sem_seg_pred = None

        if self.use_learned_depth: 
            if self.use_sem_seg:
                include_mask = np.sum(sem_seg_pred, axis=2).astype(bool).astype(float)
            else:
                include_mask = np.ones((self.DW, self.DH), dtype=float)
            include_mask = np.expand_dims(np.expand_dims(include_mask, 0), 0)
            include_mask = torch.tensor(include_mask).to(self.depth_gpu)

            depth = self.depth_pred_later(include_mask, rgb)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width/args.frame_width # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            if ds%1>0:
                depth = cv2.resize(depth, (args.frame_width, args.frame_width), interpolation = cv2.INTER_NEAREST)
            else:
                ds=int(ds)
                depth = depth[ds//2::ds, ds//2::ds]
                depth = np.expand_dims(depth, axis=2)

        return depth, sem_seg_pred

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0]*1 #shape (h,w)

        if False: #self.picked_up:
            mask_err_below = depth <0.5
            if not(self.picked_up_mask is None):
                mask_picked_up = self.picked_up_mask == 1
                depth[mask_picked_up] = 100.0
        else:
            mask_err_below = depth <0.0
        depth[mask_err_below] = 100.0
        
        depth = depth * 100
        return depth

    def depth_pred_later(self, sem_seg_pred, rgb):
        # rgb = cv2.cvtColor(self.event.frame.copy(), cv2.COLOR_RGB2BGR)#shape (h, w, 3)
        rgb = cv2.cvtColor(rgb.astype(np.uint8).copy(), cv2.COLOR_RGB2BGR)#shape (h, w, 3)
        if not (rgb.shape[-2]==300 and rgb.shape[-3]==300): # expects 300x300
            rgb = cv2.resize(rgb, (300, 300), interpolation = cv2.INTER_AREA)

        rgb_image = torch.from_numpy(rgb).permute((2, 0, 1)).unsqueeze(0).half() / 255
        self.camera_horizon = self.agent.get_agent_head_tilt()

        if abs(self.camera_horizon - 0) <5:
            _, pred_depth = self.depth_pred_model_0.predict(rgb_image.to(device=self.depth_gpu).float()) 
        elif self.camera_horizon > 29:
            _, pred_depth = self.depth_pred_model.predict(rgb_image.to(device=self.depth_gpu).float())
        else:
            depth = np.zeros((300, 300, 1))
            depth[:] = np.nan
            return depth
        if abs(self.camera_horizon - 0) <5:
            include_mask_prop=self.args.valts_trustworthy_obj_prop0
        else:
            include_mask_prop=self.args.valts_trustworthy_obj_prop
        depth_img = pred_depth.get_trustworthy_depth(max_conf_int_width_prop=self.args.valts_trustworthy_prop, include_mask=sem_seg_pred, include_mask_prop=include_mask_prop) #default is 1.0
        depth_img = depth_img.squeeze().detach().cpu().numpy()
        self.learned_depth_frame = pred_depth.depth_pred.detach().cpu().numpy()
        self.learned_depth_frame = self.learned_depth_frame.reshape((50,300,300))
        self.learned_depth_frame = 5 * 1/50 * np.argmax(self.learned_depth_frame, axis=0) #Now shape is (300,300)
        del pred_depth
        depth = depth_img

        depth = np.expand_dims(depth, 2)
        return depth

    def get_depth_map(self, rgb, head_tilt, filter_depth_by_sem=False):
        
        if self.estimate_depth:
            obs = np.zeros((self.W, self.H, 4))
            obs[:, :, :3] = rgb
            obs = obs.transpose(2, 0, 1)
            # with torch.no_grad():
            # if not (rgb.shape[0]==self.DW and rgb.shape[1]==self.DH):
            #     rgb = cv2.resize(rgb.copy(), (self.DW, self.DH), interpolation = cv2.INTER_AREA)
            # include_mask = torch.tensor(np.ones((1,1,self.DW,self.DH)).astype(bool).astype(float)).cuda()
            # depth = self.depth_pred_later(rgb, head_tilt, include_mask)
            depth, sem_seg_pred = self._preprocess_obs(obs) #, self.min_depth, self.max_depth)
            # depth = state[3,:,:]
            depth = np.squeeze(depth)
            # depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            depth /= 100. # convert to meters
            depth = depth.astype(np.float32)

            # plt.figure()
            # plt.imshow(depth)
            # plt.colorbar()
            # plt.savefig('data/images/test3.png')
            # st()
        else:
            # print("Using GT depth")
            depth = self.controller.last_event.depth_frame
            sem_seg_pred = None

        return depth, sem_seg_pred


# class Depth_ZOE():
#     def __init__(self, estimate_depth=True, task=None, on_aws=False, DH=300, DW=300):
#         self.task = task
#         self.agent = self.task

#         self.W = args.W
#         self.H = args.H

#         self.DH = DH
#         self.DW = DW

#         self.args = args

#         self.estimate_depth = estimate_depth
#         self.use_learned_depth = estimate_depth
#         self.use_sem_seg = args.use_sem_seg

#         self.res = transforms.Compose([transforms.ToPILImage(),
# 					transforms.Resize((args.frame_height, args.frame_width),
# 									  interpolation = Image.NEAREST)])

#         self.max_depth = 10.0
#         self.min_depth = 0.01

#         if self.estimate_depth:
#             repo = "isl-org/ZoeDepth"
#             # self.zoe = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
#             self.zoe = torch.hub.load(repo, "ZoeD_N", pretrained=True)
#             self.zoe.to(device) 
#             self.zoe.eval()

#             if args.zoedepth_checkpoint is not None:
#                 checkpoint = torch.load(args.zoedepth_checkpoint)
#                 checkpoint['model_state_dict'] = {k.replace("model.", ""):v for k, v in checkpoint['model_state_dict'].items()}
#                 checkpoint['model_state_dict'] = {k.replace("core.core.pretrained.", "core.core.pretrained.model."):v for k, v in checkpoint['model_state_dict'].items()}
#                 checkpoint['model_state_dict'] = {k.replace("core.core.pretrained.model.act_postprocess", "core.core.pretrained.act_postprocess"):v for k, v in checkpoint['model_state_dict'].items()}
#                 self.zoe.load_state_dict(checkpoint['model_state_dict'], strict=True)

#     def get_depth_map(self, rgb, head_tilt, filter_depth_by_sem=False):
        
#         if self.estimate_depth:
#             image_depth = Image.fromarray(rgb).convert("RGB")
#             depth = self.zoe.infer_pil(image_depth)
#             sem_seg_pred = None
#         else:
#             depth = self.task.env.simulator.controller.last_event.depth_frame
#             depth_invalid = np.logical_or(depth>self.max_depth, depth<self.min_depth)
#             depth[depth_invalid] = np.nan
#             # plt.figure()
#             # plt.imshow(depth2)
#             # plt.colorbar()
#             # plt.savefig('output/images/test.png')
#             sem_seg_pred = None

#         return depth, sem_seg_pred

class Depth_ZOE():
    def __init__(self, estimate_depth=True, task=None, on_aws=False, DH=300, DW=300):
        from nets.depthnet import DepthNet

        self.task = task
        self.agent = self.task

        self.W = args.W
        self.H = args.H

        self.DH = DH
        self.DW = DW

        self.args = args

        self.estimate_depth = estimate_depth
        self.use_learned_depth = estimate_depth
        # self.use_sem_seg = args.use_sem_seg

        # self.res = transforms.Compose([transforms.ToPILImage(),
		# 			transforms.Resize((args.frame_height, args.frame_width),
		# 							  interpolation = Image.NEAREST)])

        self.max_depth = 10.0
        self.min_depth = 0.01


        if self.estimate_depth:
            # repo = "isl-org/ZoeDepth"
            # # self.zoe = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
            # self.zoe = torch.hub.load(repo, "ZoeD_N", pretrained=True)
            # self.zoe.to(device) 
            # self.zoe.eval()

            self.model = DepthNet(pretrained=False)
            # self.model.to(device)

            if args.zoedepth_checkpoint is not None:
                checkpoint = torch.load(args.zoedepth_checkpoint)
                # checkpoint['model_state_dict'] = {k.replace("model.", ""):v for k, v in checkpoint['model_state_dict'].items()}
                # checkpoint['model_state_dict'] = {k.replace("core.core.pretrained.", "core.core.pretrained.model."):v for k, v in checkpoint['model_state_dict'].items()}
                # checkpoint['model_state_dict'] = {k.replace("core.core.pretrained.model.act_postprocess", "core.core.pretrained.act_postprocess"):v for k, v in checkpoint['model_state_dict'].items()}
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)

            self.model.to(device)
            self.model.eval()

    @torch.no_grad()
    def get_depth_map(self, rgb, head_tilt, filter_depth_by_sem=False):
        
        if self.estimate_depth:
            # image_depth = Image.fromarray(rgb).convert("RGB")
            rgb_norm = rgb.astype(np.float32) * 1./255
            rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
            # depth = self.controller.last_event.depth_frame
            # depth_torch = torch.from_numpy(depth.copy())
            depth = self.model(rgb_torch)
            depth = torch.nn.functional.interpolate(
                                    depth,
                                    size=(self.W, self.H),
                                    mode="nearest",
                                    # align_corners=False,
                                ).squeeze().cpu().numpy()

            # plt.figure()
            # plt.imshow(depth)
            # plt.colorbar()
            # plt.savefig('output/images/test.png')

            # plt.figure()
            # plt.imshow(self.task.env.simulator.controller.last_event.depth_frame)
            # plt.colorbar()
            # plt.savefig('output/images/test1.png')
            # st()

            depth_invalid = np.logical_or(depth>self.max_depth, depth<self.min_depth)
            depth[depth_invalid] = np.nan
            sem_seg_pred = None
        else:
            depth = self.task.env.simulator.controller.last_event.depth_frame
            # plt.figure()
            # plt.imshow(depth2)
            # plt.colorbar()
            # plt.savefig('output/images/test.png')
            sem_seg_pred = None

        return depth, sem_seg_pred