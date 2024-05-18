import cv2, ctypes, logging, os, numpy as np, pickle
from numpy import ma
from collections import OrderedDict
from skimage.morphology import binary_closing, disk
import scipy, skfmm
import matplotlib.pyplot as plt
import ipdb
from scipy.sparse import csr_matrix
from scipy.spatial import distance
import torch
st = ipdb.set_trace

class ShortestPathPlanner():
    def __init__(self, traversible, DT, step_size, resolution, actions_inv, obstructed_positions, locs_on_map, step_count=0):
        '''
        traversible: traversible map from explorer
        DT: Magnitude of yaw rotation in degrees
        step_size: magnitude of step size in meters
        resolution: magnitude of map resolution in meters
        '''
        self.traversible = traversible
        # self.angle_value = [0, -2.0*np.pi/num_rots, +2.0*np.pi/num_rots, 0]
        self.du = step_size
        # self.num_rots = num_rots
        self.DT = DT
        self.DT_rad = np.deg2rad(self.DT)
        # self.action_list = self.search_actions()
        # self.obstructed_actions = obstructed_actions
        self.resolution = resolution
        self.actions_inv = actions_inv
        self.obstructed_positions = obstructed_positions
        self.locs_on_map = locs_on_map       
        self.step_count = step_count 
    
    def set_goal(self, goal):
        traversible_ma = ma.masked_values(self.traversible*1, 0)
        goal_x, goal_y = int(goal[0]),int(goal[1])
        goal_x = min(goal_x, traversible_ma.shape[1]-1)
        goal_y = min(goal_y, traversible_ma.shape[0]-1)
        goal_x = max(goal_x, 0)
        goal_y = max(goal_y, 0)
        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        self.goal = goal
        return dd_mask

    def build_navigation_graph(self, state=None, last_action_success=True):
        # subsample to resolution of action space
        subsample = int(self.du/self.resolution)
        size_new_W, size_new_H = int(self.traversible.shape[0]/subsample), int(self.traversible.shape[1]/subsample)
        # traversible_sub = self.traversible[::subsample, ::subsample].astype(np.int32)
        
        # downsample map
        traversible_sub = torch.nn.functional.interpolate(
            torch.from_numpy(self.traversible.astype(np.float32)).unsqueeze(0).unsqueeze(0), 
            size=(size_new_W,size_new_H), 
            # mode='bilinear',
            mode='nearest',
            # align_corners=False,
            ).round().squeeze().numpy().astype(np.int32)

        if self.obstructed_positions is not None:
            # make any obstructed states not traversible
            if len(self.obstructed_positions)>0:
                obstructed_positions_subsample = np.int32(np.round(self.obstructed_positions/subsample))
                traversible_sub[obstructed_positions_subsample[:,1], obstructed_positions_subsample[:,0]] = 0

        if self.locs_on_map is not None:
            # make all previous states visited traversible
            locs_on_map_subsample = np.int32(np.round(self.locs_on_map/subsample))
            traversible_sub[locs_on_map_subsample[:,0], locs_on_map_subsample[:,1]] = 1

        if state is not None:
            # make current state traversible
            state_subsample = np.int32(np.round(np.asarray(state[:2])/subsample))
            traversible_sub[state_subsample[0], state_subsample[1]] = 1

        if not last_action_success:
            if len(self.obstructed_positions)>0:
                obstructed_positions_subsample = np.int32(np.round(self.obstructed_positions[-1:]/subsample))
                # print(obstructed_positions_subsample)
                traversible_sub[obstructed_positions_subsample[:,1], obstructed_positions_subsample[:,0]] = 0
        
        rows,cols = traversible_sub.shape

        n = rows*cols
        M = np.zeros((n, n))
        for r in range(rows):
            for c in range(cols):
                i = r*cols + c
                # Two inner diagonals
                if c > 0: M[i-1,i] = M[i,i-1] = 1
                # Two outer diagonals
                if r > 0: M[i-cols,i] = M[i,i-cols] = 1

        where_obstacles_j, where_obstacles_i = np.where(traversible_sub==False)
        for w in range(len(where_obstacles_i)):
            idx = where_obstacles_i[w]*cols + where_obstacles_j[w]
            M[:,idx] = 0
            M[idx,:] = 0

        # # get manhattan distance of every grid location to every other grid location
        # X1,Y1,X2,Y2 = np.ogrid[:H,:W,:H,:W]
        # d_out = np.abs(X2-X1) + np.abs(Y2-Y1)
        # # d_out = ne.evaluate('(X2-X1) + (Y2-Y1)')
        # # d_out = numpy.sqrt( (X2-X1)**2 + (Y2-Y1)**2 )
        # d_out = np.float32(d_out.reshape(H*W, H*W))

        # # get locations of obstacles
        # where_obstacles = np.where(traversible_action_resolution.reshape(H*W)==False)[0]
        # d_out[:,where_obstacles] = np.inf
        # self.graph2 = M
        M = np.ma.masked_values(M, 0)
        traversible_sparse = csr_matrix(M)
        self.graph = traversible_sparse
        self.traversible_subsample = traversible_sub
        

    def find_best_action_set(self, state, goal_closest_reachable=True, last_action_success=True):
        subsample = int(self.du/self.resolution)
        angle = state[2]
        if angle<=np.deg2rad(-180):
            angle = angle+np.deg2rad(360)
        if angle>np.deg2rad(180):
            angle = angle-np.deg2rad(360)
        

        # build navigation graph
        self.build_navigation_graph(state=state, last_action_success=last_action_success)

        state_subsample = np.int32(np.round(np.asarray(state[:2])/subsample))
        goal_subsample_original = np.int32(np.round(np.asarray(self.goal)/subsample))

        if goal_closest_reachable:
            # get reachable to agent
            traversible_ma = ma.masked_values(self.traversible_subsample*1, 0)
            goal_x, goal_y = int(state_subsample[0]),int(state_subsample[1])
            goal_x = min(goal_x, traversible_ma.shape[1]-1)
            goal_y = min(goal_y, traversible_ma.shape[0]-1)
            goal_x = max(goal_x, 0)
            goal_y = max(goal_y, 0)
            traversible_ma[goal_y, goal_x] = 0
            dd = skfmm.distance(traversible_ma, dx=1)
            reachable = np.invert(np.isnan(ma.filled(dd, np.nan)))

            # get traversible + reachable
            traversible_reachable = np.logical_and(reachable.astype(bool), self.traversible_subsample.astype(bool))

            # get closest reachable point near the goal
            inds_i, inds_j = np.where(traversible_reachable)
            reachable_where = np.stack([inds_j, inds_i], axis=0)
            dist = distance.cdist(np.expand_dims(goal_subsample_original, axis=0), reachable_where.T) # distance to goal
            # if True:
            #     # take into account distance to agent
            #     dist_to_agent = distance.cdist(np.expand_dims(state_subsample, axis=0), reachable_where.T)
            #     dist += dist_to_agent
            argmin = np.argmin(dist)
            ind_i, ind_j = inds_i[argmin], inds_j[argmin]
            goal_subsample = np.array([ind_j, ind_i])
        else:
            goal_subsample = goal_subsample_original

        rows,cols = int(self.traversible.shape[0]/subsample), int(self.traversible.shape[1]/subsample)
        state_idx = state_subsample[0]*cols + state_subsample[1]
        goal_idx = goal_subsample[0]*cols + goal_subsample[1]
        distances, predecessors = scipy.sparse.csgraph.shortest_path(csgraph=self.graph, return_predecessors=True, indices=state_idx)

        path = []
        
        i = goal_idx
        map_idxs = np.arange(self.graph.shape[0])

        # if np.all(goal_subsample==state_subsample):
        #     st()

        while i != state_idx:
            if i==-9999:
                st()
                # this means a path could not be found
                return None, None, [None], [None]
            path_step = list(np.unravel_index(map_idxs[i], (rows, cols)))
            # path_step[0] = path_step[0]*subsample
            # path_step[1] = path_step[1]*subsample
            path.append(path_step)
            i = predecessors[i]
        path.append(state_subsample)

        # plan to each location in the path
        action_set = []
        path.reverse()
        path = np.asarray(path)
        # print("Start path..")
        for i in range(len(path)-1):
            state_diff = path[i+1] - path[i]
            if state_diff[0]==-1: # need to be facing -90 degrees
                desired_angle = np.deg2rad(90)
            elif state_diff[0]==1:
                desired_angle = np.deg2rad(-90)
            elif state_diff[1]==-1:
                desired_angle = np.deg2rad(180)
            elif state_diff[1]==1:
                desired_angle = np.deg2rad(0)
            relative_yaw = angle - desired_angle
            num_rots = int(np.abs(relative_yaw/self.DT_rad))

            # if i==0:
            #     st()
            
            # rotate to correct orientatiom
            if relative_yaw<=np.deg2rad(-180):
                relative_yaw = relative_yaw+np.deg2rad(360)
            if relative_yaw>np.deg2rad(180):
                relative_yaw = relative_yaw-np.deg2rad(360)
            if relative_yaw > 0:
                yaw_action = 'RotateRight'
            elif relative_yaw < 0:
                yaw_action = 'RotateLeft'
            
            if not last_action_success and num_rots>1:
                # agent can get stuck trying to turn one direction - so try to plan the other direction
                num_rots = int(np.abs(np.deg2rad(360)/self.DT_rad)) - num_rots
                if yaw_action=='RotateRight':
                    yaw_action='RotateLeft'
                else:
                    yaw_action='RotateRight'
                for _ in range(num_rots):
                    # print(yaw_action)
                    action_set.append(self.actions_inv[yaw_action])
            else:
                for _ in range(num_rots):
                    # print(yaw_action)
                    action_set.append(self.actions_inv[yaw_action])

            # for _ in range(num_rots):
            #     # print(yaw_action)
            #     action_set.append(self.actions_inv[yaw_action])
            
            # angle should be desired angle after adjustment
            angle = desired_angle

            # move to next waypoint in the path
            # print("MoveAhead")
            action_set.append(self.actions_inv["MoveAhead"])     
        # print("End path..")     

        visualize = False #True if len(self.obstructed_positions)>0 else False
        if visualize:
            path_vis = np.array(path)
            # plt.figure(1); plt.clf()
            # plt.imshow(self.traversible, origin='lower', vmin=0, vmax=1, cmap='Greys')
            # plt.plot(path_vis[:,0]*subsample,path_vis[:,1]*subsample, '--o', color='green')
            # plt.plot(self.goal[0],self.goal[1], 'o', color='blue')
            # plt.plot(state[0],state[1], 'o', color='red')
            # plt.savefig('data/images/test1.png')

            # traversible_sub = scipy.misc.imresize(self.traversible, size=(int(self.traversible.shape[0]/subsample), int(self.traversible.shape[1]/subsample)), interp='bilinear')
            plt.figure(1); plt.clf()
            # plt.imshow(self.traversible[::subsample, ::subsample].astype(np.int32))
            plt.imshow(self.traversible_subsample, origin='lower', vmin=0, vmax=1, cmap='Greys')
            plt.plot(path_vis[:,0],path_vis[:,1], '--o', color='green', markersize=2)
            if len(self.obstructed_positions)>0:
                obstructed_positions_subsample = np.int32(np.round(self.obstructed_positions/subsample))
                plt.plot(obstructed_positions_subsample[:,0],obstructed_positions_subsample[:,1], 'o', color='yellow', markersize=2)
            locs_on_map_subsample = np.int32(np.round(self.locs_on_map/subsample))
            plt.plot(locs_on_map_subsample[:,1],locs_on_map_subsample[:,0], 'o', color='orange', markersize=2)
            plt.plot(goal_subsample[0],goal_subsample[1], 'o', color='blue', markersize=2)
            plt.plot(goal_subsample_original[0],goal_subsample_original[1], 'o', color='cyan', markersize=2)
            plt.plot(state_subsample[0],state_subsample[1], 'o', color='red', markersize=2)
            plt.colorbar()
            plt.savefig(f'data/images/planner/map_step{self.step_count}.png')
            plt.figure(1); plt.clf()
            plt.imshow(self.traversible, origin='lower', vmin=0, vmax=1, cmap='Greys')
            plt.savefig(f'data/images/planner/map_full_step{self.step_count}.png')
            st()

            # plt.figure(1); plt.clf()
            # plt.imshow(self.traversible.astype(np.int32), origin='lower', vmin=0, vmax=1, cmap='Greys')
            # plt.plot(self.goal[0],self.goal[1], 'o', color='blue')
            # plt.plot(state[0],state[1], 'o', color='red')
            # plt.savefig('data/images/test3.png')

        path[:,0] = path[:,0]*subsample
        path[:,1] = path[:,1]*subsample

        if len(action_set)>0:
            action = action_set[0]
        else:
            action = None

        return action, path, action_set, path
        

    def compare_goal(self, a, goal_dist):
        goal_dist = self.fmm_dist
        x,y,t = a
        cost_end = goal_dist[int(y), int(x)]
        dist = cost_end*1.
        if dist < self.du*1:
            return True
        return False

    def get_action(self, state, last_action_success=True):
        a, path, action_set, path = self.find_best_action_set(state, last_action_success=last_action_success)
        return a, path, action_set, path

def main():

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
            }

    from PIL import Image
    traversible = np.asarray(Image.open('../../data/traversible.png'))

    subsample = 5

    goal = [170,80]
    state = [120, 135, np.deg2rad(0)]

    # plt.figure(1); plt.clf()
    # plt.imshow(traversible[::subsample, ::subsample])
    # plt.plot(goal[0]/subsample,goal[1]/subsample, 'o', color='blue')
    # plt.plot(state[0]/subsample,state[1]/subsample, 'o', color='red')
    # plt.savefig('../../data/images/test.png')

    planner = ShortestPathPlanner(traversible, 90, 0.25, 0.05, actions_inv)
    planner.set_goal(goal)
    planner.build_navigation_graph()
    a, path, action_set = planner.find_best_action_set(state)


    path_vis = np.array(path)
    plt.figure(1); plt.clf()
    plt.imshow(traversible, origin='lower', vmin=0, vmax=1, cmap='Greys')
    plt.plot(path_vis[:,0]*subsample,path_vis[:,1]*subsample, '--o', color='green')
    plt.plot(goal[0],goal[1], 'o', color='blue')
    plt.plot(state[0],state[1], 'o', color='red')
    plt.savefig('../../data/images/test.png')
    st()


if __name__ == '__main__':
    main()
