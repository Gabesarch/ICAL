import logging
import numpy as np
import skimage
import skimage.morphology
from map_and_plan.mess import depth_utils as du
from map_and_plan.mess import rotation_utils as ru


import matplotlib.pyplot as plt
import tkinter
import matplotlib
import matplotlib.gridspec as gridspec

import ipdb
st = ipdb.set_trace


class Mapper():
    def __init__(self, C, sc, origin, map_size, resolution, max_depth=164, z_bins=[0.05,3], max_obj=50,loc_on_map_selem = skimage.morphology.disk(2), bounds=None):
        # Internal coordinate frame is X, Y into the scene, Z up.
        self.sc = sc
        self.C = C
        self.resolution = resolution
        self.max_depth = max_depth
        
        self.z_bins = z_bins
        map_sz = int(np.ceil((map_size*100)//(resolution*100)))
        self.map_sz = map_sz
        print("MAP SIZE:", map_sz)
        self.new_mapping = True
        if self.new_mapping:
            self.map = np.zeros((map_sz, map_sz, 2), dtype=np.float32)
        else:
            self.map = np.zeros((map_sz, map_sz, len(self.z_bins)+1), dtype=np.float32)
        self.semantic_map = np.zeros((map_sz, map_sz, max_obj), dtype=np.float32)
        self.loc_on_map = np.zeros((map_sz, map_sz), dtype=np.float32)
        self.blocked_on_map = np.zeros((map_sz, map_sz), dtype=np.float32)
        
        self.origin_xz = np.array([origin['x'], origin['z']])
        self.origin_map = np.array([(self.map.shape[0]-1)/2, (self.map.shape[0]-1)/2], np.float32)
        self.objects = {}
        self.loc_on_map_selem = loc_on_map_selem
        self.added_obstacles = np.ones((map_sz, map_sz), dtype=bool)

        self.map_threshold = 65 
        self.explored_threshold = 1

        self.num_boxes = 0

        self.step = 0

        self.bounds = None #bounds
    
    def _optimize_set_map_origin(self, origin_xz, resolution):
        return (origin_xz + 15)/ resolution

    def transform_to_current_frame(self, XYZ):
        R = ru.get_r_matrix([0.,0.,1.], angle=self.current_rotation)
        XYZ = np.matmul(XYZ.reshape(-1,3), R.T).reshape(XYZ.shape)
        XYZ[:,:,0] = XYZ[:,:,0] + self.current_position[0] - self.origin_xz[0] + self.origin_map[0]*self.resolution
        XYZ[:,:,1] = XYZ[:,:,1] + self.current_position[1] - self.origin_xz[1] + self.origin_map[1]*self.resolution
        return XYZ

    def update_position_on_map(self, position, rotation):
        self.current_position = np.array([position['x'], position['z']], np.float32)
        self.current_rotation = -np.deg2rad(rotation)
        x, y = self.get_position_on_map()
        
        self.loc_on_map[int(y), int(x)] = 1

    def remove_position_on_map(self, position):
        '''
        Removes a posiiton on the map that might have previously been visited
        '''
        position_ = np.array([position['x'], position['z']], np.float32)
        map_position = position_ - self.origin_xz + self.origin_map*self.resolution
        map_position = map_position / self.resolution
        x, y = map_position
        self.loc_on_map[int(y), int(x)] = 0

        self.blocked_on_map[int(y), int(x)] = 1

    def add_observation(self, position, rotation, elevation, depth, add_obs=True):

        

        
        self.update_position_on_map(position, rotation)
        if not add_obs:
            return
        d = depth*1.
        d[d > self.max_depth] = 0
        d[d < 0.02] = np.NaN
        d = d / self.sc
        XYZ1 = du.get_point_cloud_from_z(d, self.C);
        XYZ2 = du.make_geocentric(XYZ1*1, position['y'], elevation)
        XYZ3 = self.transform_to_current_frame(XYZ2)
        counts, is_valids, inds = du.bin_points(XYZ3, self.map.shape[0], self.z_bins, self.resolution)
        if self.new_mapping:
            agent_height_proj = counts[...,1:-1].sum(-1)
            all_height_proj = counts.sum(-1)
            map_cur = np.zeros_like(self.map)
            map_cur[:,:,1] = agent_height_proj/self.map_threshold
            map_cur[:,:,0] = all_height_proj/self.explored_threshold
            map_cur = np.clip(map_cur, a_min=0.0, a_max=1.0)
            self.map = np.max(np.stack([self.map, map_cur], axis=-1), axis=-1)
        else:
            self.map += counts
        
    def get_occupancy_vars(self, position, rotation, elevation, depth, global_downscaling):
        # this gets inds of each pixel in the depth image for a 3D occupancy 
        d = depth*1.
        d[d > self.max_depth] = 0
        d[d < 0.02] = np.NaN
        d = d / self.sc
        self.update_position_on_map(position, rotation)
        XYZ1 = du.get_point_cloud_from_z(d, self.C);
        XYZ2 = du.make_geocentric(XYZ1*1, position['y'], elevation)
        XYZ3 = self.transform_to_current_frame(XYZ2)
        counts2, is_valids2, inds2 = du.bin_points3D(XYZ3, self.map.shape[0]//global_downscaling, self.z_bins, self.resolution*global_downscaling)
        return counts2, is_valids2, inds2

    def _get_mask(self, obj, object_masks):
        rgb = np.array([obj.color[x] for x in 'rgb'])
        mask = object_masks == rgb
        mask = np.all(mask, 2)
        return mask

    def get_position_on_map(self):
        map_position = self.current_position - self.origin_xz + self.origin_map*self.resolution
        map_position = map_position / self.resolution
        return map_position

    def convert_xz_to_map_pos(self, xz):
        map_position = xz - self.origin_xz + self.origin_map*self.resolution
        map_position = map_position / self.resolution
        return map_position

    def get_position_on_map_from_aithor_position(self, position_origin):
        xz = np.array([position_origin['z'], position_origin['x']], np.float32)
        return self.convert_xz_to_map_pos(xz)
    
    def get_rotation_on_map(self):
        map_rotation = self.current_rotation
        return map_rotation

    def force_closed_walls(self):
        # make explicit boundaries to simulate full exploration
        if self.new_mapping:
            map = self.map[:,:,1]
            map_condensed = np.rint(map).astype(bool)
            where_obstacles = np.where(map_condensed)
            min_x, max_x = min(where_obstacles[0]), max(where_obstacles[0])
            min_y, max_y = min(where_obstacles[1]), max(where_obstacles[1])
            self.map[min_x-2:min_x+2,min_y-2:max_y+2,1] = 1
            self.map[max_x-2:max_x+2,min_y-2:max_y+2,1] = 1
            self.map[min_x-2:max_x+2, min_y-2:min_y+2,1] = 1
            self.map[min_x-2:max_x+2, max_y-2:max_y+2,1] = 1
        else:
            map = self.map[:,:,1:-1]
            map_condensed = map.sum(2) >= self.map_threshold
            where_obstacles = np.where(map_condensed)
            min_x, max_x = min(where_obstacles[0]), max(where_obstacles[0])
            min_y, max_y = min(where_obstacles[1]), max(where_obstacles[1])
            self.map[min_x-2:min_x+2,min_y-2:max_y+2,1] += self.map_threshold
            self.map[max_x-2:max_x+2,min_y-2:max_y+2,1] += self.map_threshold
            self.map[min_x-2:max_x+2, min_y-2:min_y+2,1] += self.map_threshold
            self.map[min_x-2:max_x+2, max_y-2:max_y+2,1] += self.map_threshold
            # self.map[min_x-2:min_x+2,:,1:-1] += self.map_threshold
            # self.map[max_x-2:max_x+2,:,1:-1] += self.map_threshold
            # self.map[:, min_y-2:min_y+2,1:-1] += self.map_threshold
            # self.map[:, max_y-2:max_y+2,1:-1] += self.map_threshold

    def add_obstacle_in_front_of_agent(self, selem, size_obstacle=10, pad_width=7):
        '''
        salem: dilation structure normally used to dilate the map for path planning
        '''
        # self.loc_on_map_selem
        # loc_on_map = self.loc_on_map.copy()
        # erosion_size = int(np.floor(selem.shape[0]/2))
        size_obstacle = self.loc_on_map_selem.shape[0] #- erosion_size
        # print("size_obstacle", size_obstacle)
        loc_on_map_salem_size = int(np.floor(self.loc_on_map_selem.shape[0]/2))
        # print("loc_on_map_salem_size", loc_on_map_salem_size)
        # loc_on_map_salem_size = int(size_obstacle/2)
        x, y = self.get_position_on_map()
        # print(self.current_rotation)
        if -np.deg2rad(0)==self.current_rotation:
            # plt.figure()
            # plt.imshow(self.get_traversible_map(skimage.morphology.disk(5), 1, True))
            # plt.plot(x, y, 'o')
            # plt.savefig('images/test.png')
            

            ys = [int(y+loc_on_map_salem_size+1), int(y+loc_on_map_salem_size+size_obstacle)]
            y_begin = min(ys)
            y_end = max(ys)
            xs = [int(x-np.floor(size_obstacle/2))-pad_width, int(x+np.floor(size_obstacle/2))+pad_width]
            x_begin = min(xs)
            x_end = max(xs)
        elif -np.deg2rad(90)==self.current_rotation:
            xs = [int(x+loc_on_map_salem_size+1), int(x+loc_on_map_salem_size+size_obstacle)]
            x_begin = min(xs)
            x_end = max(xs)
            ys = [int(y-np.floor(size_obstacle/2))-pad_width, int(y+np.floor(size_obstacle/2))+pad_width]
            y_begin = min(ys)
            y_end = max(ys)
        elif -np.deg2rad(180)==self.current_rotation:
            ys = [int(y-loc_on_map_salem_size-1), int(y-loc_on_map_salem_size-size_obstacle)]
            y_begin = min(ys)
            y_end = max(ys)
            xs = [int(x-np.floor(size_obstacle/2))-pad_width, int(x+np.floor(size_obstacle/2))+pad_width]
            x_begin = min(xs)
            x_end = max(xs)
        elif -np.deg2rad(270)==self.current_rotation:
            xs = [int(x-loc_on_map_salem_size-1), int(x-loc_on_map_salem_size-size_obstacle)]
            x_begin = min(xs)
            x_end = max(xs)
            ys = [int(y-np.floor(size_obstacle/2))-pad_width, int(y+np.floor(size_obstacle/2))+pad_width]
            y_begin = min(ys)
            y_end = max(ys)
        else:
            return 
            st()
            assert(False)

        self.added_obstacles[y_begin:y_end, x_begin:x_end] = False 

    def get_traversible_map(self, selem, point_count, loc_on_map_traversible):
        if self.new_mapping:
            obstacle = np.rint(self.map[:,:,1]).astype(bool)
        else:
            obstacle = np.sum(self.map[:,:,1:-1], 2) >= 100
        selem_initial = skimage.morphology.square(3)

        traversible = skimage.morphology.binary_dilation(obstacle, selem_initial) != True

        # also add in obstacles
        traversible = np.logical_and(self.added_obstacles, traversible)

        # add in blocked areas
        traversible = np.logical_and(skimage.morphology.binary_dilation(self.blocked_on_map, self.loc_on_map_selem) != True, traversible)

        # obstacle dilation
        traversible = 1 - traversible
        traversible = skimage.morphology.binary_dilation(traversible, selem) != True

        if loc_on_map_traversible:
            traversible_locs = skimage.morphology.binary_dilation(self.loc_on_map, self.loc_on_map_selem) == True 
            traversible = np.logical_or(traversible_locs, traversible)

        if self.bounds is not None:
            # limit to scene boundaries
            bounds_x = [self.bounds[0], self.bounds[1]]
            bounds_z = [self.bounds[2], self.bounds[3]]
            len_x_map = int((max(bounds_x) - min(bounds_x))/self.resolution)
            len_z_map = int((max(bounds_z) - min(bounds_z))/self.resolution)
            half_x_map = len_x_map//2
            half_z_map = len_z_map//2
            x_range = [int(self.origin_map[0]-half_x_map), int(self.origin_map[0]+half_x_map)]
            z_range = [int(self.origin_map[1]-half_z_map), int(self.origin_map[1]+half_z_map)]
            traversible[:z_range[0], :] = False
            traversible[z_range[1]:, :] = False
            traversible[:,:x_range[0]] = False
            traversible[:,x_range[1]:] = False
        return traversible
    
    def get_explored_map(self, selem, point_count):

        traversible = skimage.morphology.binary_dilation(self.loc_on_map, selem) == True 

        if self.new_mapping:
            explored = np.rint(self.map[:,:,0]).astype(bool) #np.sum(self.map, 2) >= point_count
        else:
            explored = np.sum(self.map, 2) >= point_count
        explored = np.logical_or(explored, traversible)

        if self.bounds is not None:
            # limit to scene boundaries
            bounds_x = [self.bounds[0], self.bounds[1]]
            bounds_z = [self.bounds[2], self.bounds[3]]
            len_x_map = int((max(bounds_x) - min(bounds_x))/self.resolution)
            len_z_map = int((max(bounds_z) - min(bounds_z))/self.resolution)
            half_x_map = len_x_map//2
            half_z_map = len_z_map//2
            x_range = [int(self.origin_map[0]-half_x_map), int(self.origin_map[0]+half_x_map)]
            z_range = [int(self.origin_map[1]-half_z_map), int(self.origin_map[1]+half_z_map)]
            explored[:z_range[0], :] = True
            explored[z_range[1]:, :] = True
            explored[:,:x_range[0]] = True
            explored[:,x_range[1]:] = True
        return explored
    
    def process_pickup(self, uuid):
        # Upon execution of a successful pickup action, clear out the map at
        # the current location, so that traversibility can be updated.
        import pdb; pdb.set_trace()

    def get_object_on_map(self, uuid):
        map_channel = 0
        if uuid in self.objects.keys():
            map_channel = self.objects[uuid]['channel_id']
        object_on_map = self.semantic_map[:,:,map_channel]
        return object_on_map > np.median(object_on_map[object_on_map > 0])
