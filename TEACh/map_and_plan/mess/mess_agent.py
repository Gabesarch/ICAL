import logging
import numpy as np
from ast import literal_eval
from .utils import Foo
from .explore import Explore
import torch, torchvision
from .config import objects_heuristic_detector_path, objects_do_heuristic
from PIL import Image

import ipdb
st = ipdb.set_trace

class MESSAgent():
    def __init__(self, obs, bounds=None):
        self.step_count = 0
        self.goal = Foo()
        self.parse_goal(obs.goal)
        logging.error(self.goal.description)
        
        self.rng = np.random.RandomState(0)

        self.cover = False
        if self.goal.category == 'cover' or self.goal.category == 'point_nav':
            self.explorer = Explore(obs, self.goal, bounds)
            self.cover = True                

    def parse_goal(self, goal, fig=None):
        self.goal.category = goal.metadata['category']
        self.goal.description = self.goal.category
        if self.goal.category == 'point_nav':
            self.goal.targets = goal.metadata['targets']
        else:
            self.goal.targets = ['dummy']
 
    def act(self, obs, fig=None, point_goal=None, keep_head_down=False):
        self.step_count += 1
        if self.step_count == 1:
            # # Simply testing if torch works or not 
            # device = torch.device('cuda')
            # a = torch.zeros((1, 3, 400, 600)).float().to(device)
            # b = torch.ones((1, 3, 400, 600)).float().to(device)
            # c = a+b
            # assert(torch.sum(c).detach().cpu().numpy() == 400*600*3)
            # logging.error('Torch OK')
            pass

        if self.cover:
            # action, param, is_valids, inds = self.explorer.act(obs, fig, point_goal)
            action, param = self.explorer.act(obs, fig, point_goal, keep_head_down=keep_head_down)
            return action, param#, is_valids, inds

