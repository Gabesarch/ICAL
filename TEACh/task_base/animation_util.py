import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import utils.geom
import torch
import os
from PIL import Image, ImageDraw, ImageFont
import ipdb
st = ipdb.set_trace
from arguments import args
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = args.dpi

class Animation():
    '''
    util for generating movies of the agent and TIDEE modules
    '''

    def __init__(self, W,H,navigation=None, name_to_id=None):  

        self.fig = plt.figure(1, dpi=args.dpi)
        plt.clf()

        self.W = W
        self.H = H

        self.name_to_id = name_to_id

        self.object_tracker = None
        self.camX0_T_origin = None

        self.image_plots = []

        self.navigation = navigation

    def add_text_only(self, text=None, add_map=True):

        image = np.ones((self.W,self.H, 3))*255.

        # ncols = 2

        if add_map and (self.navigation is not None):
            ncols = 2
        else:
            ncols = 1

        plt.clf()
        
        ax = []
        spec = gridspec.GridSpec(ncols=ncols, nrows=1, 
                figure=self.fig, left=0., right=1., wspace=0.05, hspace=0.5)
        ax.append(self.fig.add_subplot(spec[0, 0]))

        if add_map and (self.navigation is not None):
            ax.append(self.fig.add_subplot(spec[0, 1]))

        for a in ax:
            a.axis('off')

        t_i = 1
        
        for t_ in text:
            thickness=1; fontScale=0.5; color=(0,0,0)
            image = cv2.putText(np.float32(image), t_, (int(20), int(20*t_i)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
            t_i += 1

        image = image.astype(np.uint8)
        

        if ncols>1:
            ax[0].set_title("")
            
            ax[0].imshow(image)

            if add_map and self.navigation is not None:
                # plt.subplot(1,2,2)
                m_vis = np.invert(self.navigation.explorer.mapper.get_traversible_map(
                    self.navigation.explorer.selem, 1,loc_on_map_traversible=True))

                ax[1].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                        cmap='Greys')
                state_xy = self.navigation.explorer.mapper.get_position_on_map()
                state_theta = self.navigation.explorer.mapper.get_rotation_on_map()
                arrow_len = 2.0/self.navigation.explorer.mapper.resolution
                ax[1].arrow(state_xy[0], state_xy[1], 
                            arrow_len*np.cos(state_theta+np.pi/2),
                            arrow_len*np.sin(state_theta+np.pi/2), 
                            color='b', head_width=20)

                if self.navigation.explorer.point_goal is not None:
                    ax[1].plot(self.navigation.explorer.point_goal[1], self.navigation.explorer.point_goal[0], color='blue', marker='o',linewidth=10, markersize=12)

                if self.object_tracker is not None:
                    centroids, labels = self.object_tracker.get_centroids_and_labels()
                    if not isinstance(centroids, int):
                        if len(centroids)>0:
                            if centroids.shape[1]>0:
                                cmap = matplotlib.cm.get_cmap('gist_rainbow')
                                obj_center_camX0 = centroids #utils.geom.apply_4x4(self.camX0_T_origin.float(), torch.from_numpy(centroids).unsqueeze(0).float()).squeeze(0).numpy()
                                for o in range(len(obj_center_camX0)):
                                    label = labels[o]
                                    if label not in self.name_to_id.keys():
                                        continue
                                    color_id = self.name_to_id[label]/len(self.name_to_id)
                                    color = cmap(color_id)
                                    obj_center_camX0_ = {'x':obj_center_camX0[o][0], 'y':obj_center_camX0[o][1], 'z':obj_center_camX0[o][2]}
                                    map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
                                    ax[1].plot(map_pos[1], map_pos[0], color=color, marker='o',linewidth=1, markersize=4)

                ax[1].set_title("Semantic Map")

            canvas = FigureCanvas(plt.gcf())

            canvas.draw()       # draw the canvas, cache the renderer
            width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        self.image_plots.append(image)



    def add_frame(self, image, text=None, add_map=True, box=None, embedded_image=None, map_vis=None):

        if add_map and (self.navigation is not None or map_vis is not None):
            ncols = 2
        else:
            ncols = 1

        plt.clf()
        
        ax = []
        spec = gridspec.GridSpec(ncols=ncols, nrows=1, 
                figure=self.fig, left=0., right=1., wspace=0.05, hspace=0.5)
        ax.append(self.fig.add_subplot(spec[0, 0]))
        if add_map and (self.navigation is not None or map_vis is not None):
            ax.append(self.fig.add_subplot(spec[0, 1]))

        for a in ax:
            a.axis('off')

        if box is not None:
            rect_th=1; text_size=text_size; text_th=1
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0, 255, 0), rect_th)

        image = image.astype(np.uint8)

        if ncols>1:
            
            if text is not None:
                text_size=0.5; text_th=1
                ax[0].set_title(text)

            ax[0].imshow(image)

            if embedded_image is not None:
                newax = plt.axes([0.13,0.03,0.35,0.35], anchor='NE', zorder=1)
                newax.set_xticks([])
                newax.set_yticks([])
                newax.imshow(embedded_image.astype(np.uint8))

            if add_map and map_vis is not None:
                ax[1].imshow(map_vis)
                ax[1].set_title("Semantic Map")

            elif add_map and self.navigation is not None:
                m_vis = np.invert(self.navigation.explorer.mapper.get_traversible_map(
                    self.navigation.explorer.selem, 1,loc_on_map_traversible=True))

                ax[1].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                        cmap='Greys')
                state_xy = self.navigation.explorer.mapper.get_position_on_map()
                state_theta = self.navigation.explorer.mapper.get_rotation_on_map()
                arrow_len = 2.0/self.navigation.explorer.mapper.resolution
                ax[1].arrow(state_xy[0], state_xy[1], 
                            arrow_len*np.cos(state_theta+np.pi/2),
                            arrow_len*np.sin(state_theta+np.pi/2), 
                            color='b', head_width=20)

                if self.navigation.explorer.point_goal is not None:
                    ax[1].plot(self.navigation.explorer.point_goal[1], self.navigation.explorer.point_goal[0], color='blue', marker='o',linewidth=10, markersize=12)

                if self.object_tracker is not None:
                    centroids, labels = self.object_tracker.get_centroids_and_labels()
                    if not isinstance(centroids, int):
                        if len(centroids)>0:
                            if centroids.shape[1]>0:
                                cmap = matplotlib.cm.get_cmap('gist_rainbow')
                                obj_center_camX0 = centroids #utils.geom.apply_4x4(self.camX0_T_origin.float(), torch.from_numpy(centroids).unsqueeze(0).float()).squeeze(0).numpy()
                                for o in range(len(obj_center_camX0)):
                                    label = labels[o]
                                    if label not in self.name_to_id.keys():
                                        continue
                                    color_id = self.name_to_id[label]/len(self.name_to_id)
                                    color = cmap(color_id)
                                    obj_center_camX0_ = {'x':obj_center_camX0[o][0], 'y':obj_center_camX0[o][1], 'z':obj_center_camX0[o][2]}
                                    map_pos = self.navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)
                                    ax[1].plot(map_pos[1], map_pos[0], color=color, marker='o',linewidth=1, markersize=4)



                ax[1].set_title("Semantic Map")

            canvas = FigureCanvas(plt.gcf())

            canvas.draw()       # draw the canvas, cache the renderer
            width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        else:
            if text is not None:
                text_size=0.5; text_th=1
                cv2.putText(image,text,(int(20), int(20)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=text_th)

        self.image_plots.append(image)

    def render_movie(self, dir,  episode, tag='', fps=5):
        if not os.path.exists(dir):
            os.mkdir(dir)
        video_name = os.path.join(dir, f'output{episode}_{tag}.mp4')
        print(f"rendering to {video_name}")
        height, width, _ = self.image_plots[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

        for im in self.image_plots:
            rgb = np.array(im).astype(np.uint8)
            bgr = rgb[:,:,[2,1,0]]
            video_writer.write(bgr)

        cv2.destroyAllWindows()
        video_writer.release()



    