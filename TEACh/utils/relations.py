import numpy as np
import hyperparams as hyp
import re
import ipdb
st = ipdb.set_trace


def extract_relations_centroids(centroid_target, label_target, obj_centroids, obj_labels_np, floor_height=0.0, pos_translator=None, overhead_map=None): 

    '''Extract relationships of interest from a list of objects'''
    '''
    centroid_target: subject centroid to extract relations for
    label_target: semantic label of centroid_target
    obj_centroids: object centroids to compare to subject
    obj_labels_np: semantic labels of obj_centroids as numpy array
    '''

    # check if centroid exists in obj_centroids to remove it
    check = np.linalg.norm(obj_centroids - np.expand_dims(centroid_target, 0), axis=1)
    check = check > 1e-3
    obj_centroids = obj_centroids[check]
    obj_labels_np = obj_labels_np[check]
    # obj_labels_np = np.array(obj_labels.copy())

    obj_labels = list(obj_labels_np)

    # centroid_targ

    ################# Check Relationships #################
    # check pairwise relationships. this loop is order agnostic, since pairwise relationships are mostly invertible
    if hyp.visualize_relations:
        relations_dict = {}
        for relation in relations_executors_pairs:
            relations_dict[relation] = []
    relations = []
    for relation in relations_executors_pairs:
        relation_fun = relations_executors_pairs[relation]
        if relation=='closest-to' or relation=='farthest-to' or relation=='supported-by':
            if relation=='supported-by':
                if label_target in receptacles:
                    continue
                yes_recept = []
                for obj_label_i in obj_labels:
                    if obj_label_i in receptacles:
                        yes_recept.append(True)
                    else:
                        yes_recept.append(False)
                yes_recept = np.array(yes_recept)
                obj_centroids_ = obj_centroids[yes_recept]
                obj_labels_ = list(obj_labels_np[yes_recept])
                relation_ind = relation_fun(centroid_target, obj_centroids_, ground_plane_h=floor_height)
                if relation_ind==-2:
                    pass
                elif relation_ind==-1:
                    relations.append("The {0} is {1} the {2}".format(format_class_name(label_target), relation.replace('-', ' '), format_class_name('Floor')))
                    if hyp.visualize_relations:
                        relations_dict[relation].append(centroid_target)
                else:
                    relations.append("The {0} is {1} the {2}".format(format_class_name(label_target), relation.replace('-', ' '), format_class_name(obj_labels_[relation_ind])))
                    if hyp.visualize_relations:
                        relations_dict[relation].append(obj_centroids_[relation_ind])

            else:
                relation_ind = relation_fun(centroid_target, obj_centroids)
                if relation_ind==-2:
                    pass
                elif relation_ind==-1:
                    relations.append("The {0} is {1} the {2}".format(format_class_name(label_target), relation.replace('-', ' '), format_class_name('Floor')))
                    if hyp.visualize_relations:
                        relations_dict[relation].append(centroid_target)
                else:
                    relations.append("The {0} is {1} the {2}".format(format_class_name(label_target), relation.replace('-', ' '), format_class_name(obj_labels[relation_ind])))
                    if hyp.visualize_relations:
                        relations_dict[relation].append(obj_centroids[relation_ind])
        else:
            for i in range(len(obj_centroids)):

                is_relation = relation_fun(centroid_target, obj_centroids[i])
            
                if is_relation:
                    relations.append("The {0} is {1} the {2}".format(format_class_name(label_target), relation.replace('-', ' '), format_class_name(obj_labels[i])))
                    if hyp.visualize_relations:
                        relations_dict[relation].append(obj_centroids[i])

    if False:
        colors_rels = {
        'next-to': (0, 255, 0),
        'supported-by': (0, 255, 0),
        'closest-to': (0, 255, 255)
        }
        img = overhead_map.copy()

        c_target = pos_translator(centroid_target)
        color = (255, 0, 0)
        thickness = 1
        cv2.circle(img, c_target[[1,0]], 7, color, thickness)
        radius = 5
        for relation in list(relations_dict.keys()):
            centers_relation = relations_dict[relation]
            color = colors_rels[relation]
            for c_i in range(len(centers_relation)):
                center_r = centers_relation[c_i]
                c_rel_im = pos_translator(center_r)
                cv2.circle(img, c_rel_im[[1,0]], radius, color, thickness)

        plt.figure(figsize=(8,8))
        plt.imshow(img)
        plt.savefig('images/test.png')
        st()

    return relations

def format_class_name(name):
    try:
        if name=="TVStand":
            formatted = "television stand"
        elif name=="CounterTop":
            formatted = "countertop"
        else:
            formatted = re.sub(r"(?<=\w)([A-Z])", r" \1", name).lower()
    except:
        st()
    return formatted

# @staticmethod
# def box2points(box):
#     """Convert box min/max coordinates to vertices (8x3)."""
#     x_min, y_min, z_min, x_max, y_max, z_max = box
#     return np.array([
#         [x_min, y_min, z_min], [x_min, y_max, z_min],
#         [x_max, y_min, z_min], [x_max, y_max, z_min],
#         [x_min, y_min, z_max], [x_min, y_max, z_max],
#         [x_max, y_min, z_max], [x_max, y_max, z_max]
#     ])

# @staticmethod
# def _compute_dist(points0, points1):
#     """Compute minimum distance between two sets of points."""
#     dists = ((points0[:, None, :] - points1[None, :, :]) ** 2).sum(2)
#     return dists.min()

# def _intersect(box_a, box_b):
#     return _intersection_vol(box_a, box_b) > 0

# @staticmethod
# def _intersection_vol(box_a, box_b):
#     xA = max(box_a[0] - box_a[3] / 2, box_b[0] - box_b[3] / 2)
#     yA = max(box_a[1] - box_a[4] / 2, box_b[1] - box_b[4] / 2)
#     zA = max(box_a[2] - box_a[5] / 2, box_b[2] - box_b[5] / 2)
#     xB = min(box_a[0] + box_a[3] / 2, box_b[0] + box_b[3] / 2)
#     yB = min(box_a[1] + box_a[4] / 2, box_b[1] + box_b[4] / 2)
#     zB = min(box_a[2] + box_a[5] / 2, box_b[2] + box_b[5] / 2)
#     return max(0, xB - xA) * max(0, yB - yA) * max(0, zB - zA)

# def _inside(box_a, box_b):
#     volume_a = box_a[3] * box_a[4] * box_a[5]
#     return np.isclose(_intersection_vol(box_a, box_b), volume_a)

# @staticmethod
# def iou_2d(box0, box1):
#     """Compute 2d IoU for two boxes in coordinate format."""
#     box_a = np.array(box0)[(0, 1, 3, 4), ]
#     box_b = np.array(box1)[(0, 1, 3, 4), ]
#     # Intersection
#     xA = max(box_a[0], box_b[0])
#     yA = max(box_a[1], box_b[1])
#     xB = min(box_a[2], box_b[2])
#     yB = min(box_a[3], box_b[3])
#     inter_area = max(0, xB - xA) * max(0, yB - yA)
#     # Areas
#     box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
#     box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
#     # Return IoU and area ratios
#     return (
#         inter_area / (box_a_area + box_b_area - inter_area),  # iou
#         [inter_area / box_a_area, inter_area / box_b_area],
#         [box_a_area / box_b_area, box_b_area / box_a_area]
#     )

# @staticmethod
# def volume(box):
#     """Compute of box in coordinate format (min, max)."""
#     return (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2])

# @staticmethod
# def _to_min_max(box):
#     return np.concatenate((
#         box[:, :3] - box[:, 3:] / 2, box[:, :3] + box[:, 3:] / 2
#     ), 1)

# @staticmethod
# def _same_x_range(box, ref_box):
#     return (
#         min(box[3], ref_box[3]) - max(box[0], ref_box[0])
#         > 0.8 * min(box[3] - ref_box[0], box[3] - ref_box[0])
#     )

# @staticmethod
# def _same_y_range(box, ref_box):
#     return (
#         min(box[4], ref_box[4]) - max(box[1], ref_box[1])
#         > 0.8 * min(box[4] - ref_box[1], box[4] - ref_box[1])
#     )

# @staticmethod
# def _same_z_range(box, ref_box):
#     return (
#         min(box[5], ref_box[5]) - max(box[2], ref_box[2])
#         > 0.3 * (box[5] - box[2])
#     )

# def _is_left(box, ref_box):
#     return (
#         box[3] < ref_box[0]  # x_max < x_ref_min
#         and _same_y_range(box, ref_box)
#         and _same_z_range(box, ref_box)
#     )

# def _is_right(box, ref_box):
#     return (
#         box[0] > ref_box[3]  # x_min > x_ref_max
#         and _same_y_range(box, ref_box)
#         and _same_z_range(box, ref_box)
#     )

# def _is_front(box, ref_box):
#     return (
#         box[4] < ref_box[1]  # y_max < y_ref_min
#         and _same_x_range(box, ref_box)
#         and _same_z_range(box, ref_box)
#     )

# def _is_behind(box, ref_box):
#     return (
#         box[1] > ref_box[4]  # y_min > y_ref_max
#         and _same_x_range(box, ref_box)
#         and _same_z_range(box, ref_box)
#     )

# # def _is_next_to(box, ref_box):
# #     center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2])
# #     center_ref_box0 = np.array([(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2, (ref_box[5]+ref_box[2])/2])
# #     squared_dist = np.sum((center_box-center_ref_box0)**2, axis=0)
# #     dist = np.sqrt(squared_dist)

# #     return (
# #         dist < 2.0  
# #         and _same_z_range(box, ref_box)
# #     )

# def _is_between(box, ref_box0, ref_box1):
#     # # Get the convex hull of all points of the two anchors
#     # convex_hull = MultiPoint(
#     #     tuple(map(tuple, box2points(ref_box0)[:4, :2]))
#     #     + tuple(map(tuple, box2points(ref_box1)[:4, :2]))
#     # ).convex_hull
#     # # Get box as polygons
#     # polygon_t = MultiPoint(
#     #     tuple(map(tuple, box2points(box)[:4, :2]))
#     # ).convex_hull
#     # # Candidate should fall in the convex_hull polygon
#     # return (
#     #     convex_hull.intersection(polygon_t).area / polygon_t.area > 0.51
#     #     and _same_z_range(box, ref_box0)
#     #     and _same_z_range(box, ref_box1)
#     # )

#     # Find distance to of the center of the box to line between centers of ref_box0 and ref_box1

#     # get centers of each box
#     center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2])
#     center_ref_box0 = np.array([(ref_box0[3]+ref_box0[0])/2, (ref_box0[4]+ref_box0[1])/2, (ref_box0[5]+ref_box0[2])/2])
#     center_ref_box1 = np.array([(ref_box1[3]+ref_box1[0])/2, (ref_box1[4]+ref_box1[1])/2, (ref_box1[5]+ref_box1[2])/2])

#     dist_to_0 = np.sqrt(np.sum((center_box - center_ref_box0)**2, axis=0))
#     dist_to_1 = np.sqrt(np.sum((center_box - center_ref_box1)**2, axis=0))

#     # # draw line between center_ref_box0 and center_ref_box1, and find distance of center_box to that line
#     # p = center_ref_box0
#     # q = center_ref_box1
#     # r = center_box
#     # x = p-q
#     # t = np.dot(r-q, x)/np.dot(x, x)
#     # dist = np.linalg.norm(t*(p-q)+q-r)
#     # print(dist)

#     a = center_ref_box0
#     b = center_ref_box1
#     p = center_box

#     # # normalized tangent vector
#     # d = np.divide(b - a, np.linalg.norm(b - a))

#     # # signed parallel distance components
#     # s = np.dot(a - p, d)
#     # t = np.dot(p - b, d)

#     # # clamped parallel distance
#     # h = np.maximum.reduce([s, t, 0])

#     # # perpendicular distance component
#     # c = np.cross(p - a, d)

#     # dist = np.hypot(h, np.linalg.norm(c))
#     # print(dist)

#     r = b - a
#     a2 = a - p
    
#     min_t = np.clip(-a2.dot(r) / (r.dot(r)), 0, 1)
#     d = a2 + min_t * r
#     dist = np.sqrt(d.dot(d))
#     # print(dist)

#     # distance to each center
#     squared_dist = np.sum((p-a)**2, axis=0)
#     dist3 = np.sqrt(squared_dist)
#     # print(dist3)

#     squared_dist = np.sum((p-b)**2, axis=0)
#     dist4 = np.sqrt(squared_dist)
#     # print(dist4)

#     closer_to_point_than_segment = np.logical_or(np.isclose(dist, dist3), np.isclose(dist, dist4))

#     # # check if not to the right or left of both ref boxes
#     # is_right_0 = _is_right(box, ref_box0)
#     # is_right_1 = _is_right(box, ref_box1)
#     # is_left_0 = _is_left(box, ref_box0)
#     # is_left_1 = _is_left(box, ref_box1)
#     # both_right = is_right_0 and is_right_1
#     # both_left = is_left_0 and is_left_1
#     # # sign_diff_x = np.sign(center_box[0] - center_ref_box0[0]) != np.sign(center_box[0] - center_ref_box1[0])
#     # # sign_diff_y = np.sign(center_box[1] - center_ref_box0[1]) != np.sign(center_box[1] - center_ref_box1[1])
#     # print(not both_right, not both_left)

#     return dist, (
#         dist < 0.1 #0.1 
#         and not closer_to_point_than_segment
#         and _same_z_range(box, ref_box0)
#         and _same_z_range(box, ref_box1)
#         and not _is_above(box, ref_box0)
#         and not _is_below(box, ref_box0)
#         and not _is_above(box, ref_box1)
#         and not _is_below(box, ref_box1)
#         and max(dist_to_0, dist_to_1) < 3
#     )


# def _is_similar_dist_to(box, ref_box0, ref_box1):
#     center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2])
#     center_ref_box0 = np.array([(ref_box0[3]+ref_box0[0])/2, (ref_box0[4]+ref_box0[1])/2, (ref_box0[5]+ref_box0[2])/2])
#     center_ref_box1 = np.array([(ref_box1[3]+ref_box1[0])/2, (ref_box1[4]+ref_box1[1])/2, (ref_box1[5]+ref_box1[2])/2])

#     dist_to_0 = np.sqrt(np.sum((center_box - center_ref_box0)**2, axis=0))
#     dist_to_1 = np.sqrt(np.sum((center_box - center_ref_box1)**2, axis=0))

#     return np.abs(dist_to_0 - dist_to_1), np.abs(dist_to_0 - dist_to_1) < min(0.1, 0.1 * min(dist_to_0, dist_to_1)) and max(dist_to_0, dist_to_1) < 2

# def _is_very_close_to(box, ref_box):
#     '''
#     this is a simpler version of _is_next_to.
#     it doesn't take into account the intersection ratio. Just finds the closest distance between the points in the box
#     this is solely for the superstructure: object type B's around object type A.
#     '''
#     corner_idx = np.array([[0,1,2],[0,1,5],[0,4,2],[0,4,5],[3,1,2],[3,1,5],[3,4,2],[3,4,5]])
#     box_corners = box[corner_idx]
#     box_center = np.array([[(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2]])
#     box_points = np.concatenate([box_corners, box_center], axis=0)
#     ref_box_corners = ref_box[corner_idx]
#     ref_box_center = np.array([[(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2, (ref_box[5]+ref_box[2])/2]])
#     ref_box_points = np.concatenate([ref_box_corners, ref_box_center], axis=0)
#     squared_dists = np.sum((np.expand_dims(box_points, axis=0) - np.expand_dims(ref_box_points, axis=1))**2, axis=-1)
#     min_dist = np.amin(np.sqrt(squared_dists))
#     return dist < 1.0


# def _is_next_to(box, ref_box, ref_is_wall):
#     corner_idx = np.array([[0,1,2],[0,1,5],[0,4,2],[0,4,5],[3,1,2],[3,1,5],[3,4,2],[3,4,5]])
#     box_corners = box[corner_idx]
#     box_center = np.array([[(box[3]+box[0])/2, (box[4]+box[1])/2, (box[5]+box[2])/2]])
#     box_points = np.concatenate([box_corners, box_center], axis=0)
#     ref_box_corners = ref_box[corner_idx]
#     ref_box_center = np.array([[(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2, (ref_box[5]+ref_box[2])/2]])
#     ref_box_points = np.concatenate([ref_box_corners, ref_box_center], axis=0)
#     if not ref_is_wall:
#         squared_dists = np.sum((np.expand_dims(box_points, axis=0) - np.expand_dims(ref_box_points, axis=1))**2, axis=-1)
#         dist = np.amin(np.sqrt(squared_dists))

#         iou, intersect_ratios, area_ratios = iou_2d(box, ref_box)
#         int2box_ratio, int2ref_ratio = intersect_ratios

#         return (
#             dist < 0.5
#             and int2box_ratio < 0.7
#             and int2ref_ratio < 0.7
#             # and _is_equal_height(box, ref_box)
#             # and not _is_above(box, ref_box)
#             # and not _is_below(box, ref_box)
#         )
#     else:
#         dist = np.amin(np.abs(box_points[:,0] - ref_box_center[0,0])) if ref_box[3] - ref_box[0] < ref_box[4] - ref_box[1] else np.amin(np.abs(box_points[:,1] - ref_box_center[0,1]))
#         return dist < 0.3

# def _is_supported_by(box, ref_box, ground_plane_h=None):

#     if ground_plane_h is not None:
#         box_bottom_ref_top_dist = box[2] - ground_plane_h
#         # print(box_bottom_ref_top_dist)
#     else:
#         box_bottom_ref_top_dist = box[2] - ref_box[5]
#     iou, intersect_ratios, area_ratios = iou_2d(box, ref_box)
#     int2box_ratio, _ = intersect_ratios
#     box2ref_ratio, _ = area_ratios

#     return (
#         int2box_ratio > 0.1  #0.3  # xy intersection
#         and abs(box_bottom_ref_top_dist) <= 0.2  # close to surface
#         and box2ref_ratio < 1.5  # supporter is usually larger
#     )

# # def _is_supporting(box, ref_box):
# #     ref_bottom_cox_top_dist = ref_box[2] - box[5]
# #     _, intersect_ratios, area_ratios = iou_2d(box, ref_box)
# #     _, int2ref_ratio = intersect_ratios
# #     _, ref2box_ratio = area_ratios
# #     # print(int2ref_ratio, abs(ref_bottom_cox_top_dist), ref2box_ratio)
# #     return (
# #         int2ref_ratio > 0.1 #0.3  # xy intersection
# #         and abs(ref_bottom_cox_top_dist) <= 0.2 #0.01  # close to surface
# #         and ref2box_ratio < 1.5  # supporter is usually larger
# #     )

# def _is_above(box, ref_box):
#     box_bottom_ref_top_dist = box[2] - ref_box[5]
#     _, intersect_ratios, _ = iou_2d(box, ref_box)
#     int2box_ratio, int2ref_ratio = intersect_ratios
#     return (
#         box_bottom_ref_top_dist > 0.03  # should be above
#         and max(int2box_ratio, int2ref_ratio) > 0.2  # xy intersection
#     )

# def _is_below(box, ref_box):
#     ref_bottom_cox_top_dist = ref_box[2] - box[5]
#     _, intersect_ratios, _ = iou_2d(box, ref_box)
#     int2box_ratio, int2ref_ratio = intersect_ratios
#     return (
#         ref_bottom_cox_top_dist > 0.03  # should be above
#         and max(int2box_ratio, int2ref_ratio) > 0.2  # xy intersection
#     )

# def _is_aligned(ori, ref_ori, box, ref_box):

#     if ori is None or ref_ori is None:
#         return False

#     ori_dist = np.linalg.norm(ori[:2] - ref_ori[:2])
#     # print(ori_dist, ori, ref_ori)

#     center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2])
#     center_ref_box0 = np.array([(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2])
#     squared_dist = np.sum((center_box-center_ref_box0)**2, axis=0)
#     dist = np.sqrt(squared_dist)        

#     return (
#         np.abs(ori_dist) < 0.2
#         and dist < 1.5
#     )

# def _is_facing(overhead_box, ref_facing, overhead_box_ref, box, ref_box):
#     # algorithm from https://gamedev.stackexchange.com/questions/109513/how-to-find-if-an-object-is-facing-another-object-given-position-and-direction-a

#     if ref_facing is None or overhead_box is None:
#         return False

#     center = np.array([(overhead_box[0] + overhead_box[2])//2, (overhead_box[1] + overhead_box[3])//2])
#     center_ref = np.array([(overhead_box_ref[0] + overhead_box_ref[2])//2, (overhead_box_ref[1] + overhead_box_ref[3])//2])

#     ref_facing_unit_vec = np.array([np.cos(ref_facing-np.pi/2), np.sin(ref_facing-np.pi/2)])
#     facing = -np.dot((center_ref-center)/np.linalg.norm(center_ref-center), ref_facing_unit_vec/np.linalg.norm(ref_facing_unit_vec))

#     center_box = np.array([(box[3]+box[0])/2, (box[4]+box[1])/2])
#     center_ref_box0 = np.array([(ref_box[3]+ref_box[0])/2, (ref_box[4]+ref_box[1])/2])
#     squared_dist = np.sum((center_box-center_ref_box0)**2, axis=0)
#     dist = np.sqrt(squared_dist)    

#     # print(dist)
#     # print(facing)

#     return (
#         facing > 0.9 # 0.9
#         and dist < 7.0 # distance in meters
#     )


# def get_facing_dir(depth_masked, mask_binary, line):
#     ''' 
#     TODO: Optimize the algorithm here
#     Obtains the facing direciton with respect to the overhead map.
#     For reference: 0 degrees is facing directly "up"/north, 90 degrees is eat, 180 degrees south, etc. 
#     looks at the depth distribution and identifies the "back" of the object by which direction has a larger median depth (wrt a fitted line)
#     Inputs: 
#     - depth_masked: depth map of overhead view masked for the object
#     - mask_binary: binary mask of the object overhead view
#     - line: 4x1 array: with parameters of line from cv2.fitLine
#     '''

#     vx,vy,x,y = list(line)
    
#     x_mask, y_mask = np.where(mask_binary)

#     for q in [1,2]: # also consider perpendicular line

#         # here we need to check the original line and the line perpendicular to that
#         if q == 1:
#                 # get two points on the fitted line
#             lx1 = x + vx*256
#             ly1 = y + vy*256
#             lx2 = x + vx*0
#             ly2 = y + vy*0
#         elif q==2:
#             lx1 = x + vy*256
#             ly1 = y - vx*256
#             lx2 = x + vy*0
#             ly2 = y - vx*0

#         depths_above = []
#         depths_below = []
#         xs_above = []
#         ys_above = []
#         xs_below = []
#         ys_below = []
#         for i in range(depth_masked.shape[0]):

#             # get image coords of mask 
#             yA = x_mask[i]
#             xA = y_mask[i]

#             # Check if the point falls above or below the fitted line
#             v1 = (lx2-lx1, ly2-ly1)   # Vector 1
#             v2 = (lx2-xA, ly2-yA)   # Vector 1
#             xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product
#             if xp > 0: # fell 'above' the line (may not be in the literal sense here)
#                 depths_above.append(depth_masked[i])
#                 xs_above.append(xA)
#                 ys_above.append(yA)
#                 # print('on one side')
#             elif xp < 0: # fell 'below' the line
#                 depths_below.append(depth_masked[i])
#                 xs_below.append(xA)
#                 ys_below.append(yA)
#                 # print('on the other')
#             # else:
#             #     print('on the same line!')
        
#         if q==1:
#             # get median depth of points that fall on either side of line
#             d_ab_median1 = np.median(np.array(depths_above))
#             d_be_median1 = np.median(np.array(depths_below))

#             # want largest difference (back of chair vs front)
#             median_diff1 = np.abs(d_ab_median1 - d_be_median1)
#         elif q==2:
#             # get median depth of points that fall on either side of line
#             d_ab_median2 = np.median(np.array(depths_above))
#             d_be_median2 = np.median(np.array(depths_below))

#             # want largest difference (back of chair vs front)
#             median_diff2 = np.abs(d_ab_median2 - d_be_median2)
        
#             if median_diff2 > median_diff1: # we want bigger difference in median depth to detect the back vs front
#                 line_consider = 'perpendicular'

#                 # find distance to y axis
#                 y_axis      = np.array([0, 1])    # unit vector in the same direction as the x axis
#                 your_line   = np.array([vx, vy])  # unit vector in the same direction as your line
#                 dot_product = np.dot(y_axis, your_line)
#                 angle_2_y   = np.arccos(dot_product)

#                 # if angle_2_y > np.pi/2:
#                 #     angle_2_y = np.pi - angle_2_y

#                 # if angle_2_y > np.pi/4:
#                 #     angle_2_y = np.pi/2 - angle_2_y

#                 # get sign of slope
#                 sign_slope = np.sign(vx/vy)

#                 if d_ab_median2 < d_be_median2: # shorter distance to camera means higher depth of part from floor
#                     if sign_slope > 0:
#                         facing_orientation = np.pi - angle_2_y
#                     else:
#                         facing_orientation = 2*np.pi - angle_2_y
#                     # side_consider = 'above'
#                 else:
#                     if sign_slope > 0:
#                         facing_orientation = np.pi - angle_2_y
#                         # facing_orientation = np.pi - angle_2_y
#                     else:
#                         facing_orientation = 2*np.pi - angle_2_y
#                     # side_consider = 'below'
#             else:

#                 # find angle to y axis
#                 y_axis      = np.array([0, 1])    # unit vector in the same direction as the x axis
#                 your_line   = np.array([-vy, vx])  # unit vector in the same direction as your line
#                 dot_product = np.dot(y_axis, your_line)
#                 angle_2_y   = np.arccos(dot_product)

#                 # if angle_2_y > np.pi/2:
#                 #     angle_2_y = np.pi - angle_2_y

#                 # if angle_2_y > np.pi/4:
#                 #     angle_2_y = np.pi/2 - angle_2_y

#                 # get sign of slope
#                 sign_slope = np.sign(-vy/vx)

#                 if d_ab_median1 < d_be_median1: # shorter distance to camera means higher depth of part from floor
#                     if sign_slope > 0:
#                         facing_orientation = 2*np.pi - angle_2_y # f
#                     else:
#                         facing_orientation = 2*np.pi - angle_2_y # f
#                     # side_consider = 'above'
#                 else:
#                     if sign_slope > 0:
#                         facing_orientation = np.pi - angle_2_y
#                         # facing_orientation = np.pi - angle_2_y
#                     else:
#                         facing_orientation = 2*np.pi - angle_2_y # f

#     return facing_orientation

# @staticmethod
# def _is_higher(box, ref_box):
#     return box[2] - ref_box[5] > 0.1 #0.03

# @staticmethod
# def _is_lower(box, ref_box):
#     return ref_box[2] - box[5] > 0.1 #0.03

# def _is_equal_height(box, ref_box):
#     return min(box[5], ref_box[5]) - max(box[2], ref_box[2]) > 0.3 * (max(box[5], ref_box[5]) - min(box[2], ref_box[2]))

# def _is_larger(box, ref_box):
#     return volume(box) > 1.1 * volume(ref_box)

# def _is_smaller(box, ref_box):
#     return volume(ref_box) > 1.1 * volume(box)

# def _is_equal_size(box, ref_box):
#     return (
#         not _is_larger(box, ref_box)
#         and not _is_smaller(box, ref_box)
#         and 0.9 < (box[3] - box[0]) / (ref_box[3] - ref_box[0]) < 1.1
#         and 0.9 < (box[4] - box[1]) / (ref_box[4] - ref_box[1]) < 1.1
#         and 0.9 < (box[5] - box[2]) / (ref_box[5] - ref_box[2]) < 1.1
#     )

# def _get_closest(boxes, ref_box):
#     dists = np.array([
#         _compute_dist(box2points(box), box2points(ref_box))
#         for box in boxes
#     ])
#     # dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
#     return dists.argmin()

# def _get_furthest(boxes, ref_box):
#     dists = np.array([
#         _compute_dist(box2points(box), box2points(ref_box))
#         for box in boxes
#     ])
#     # dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
#     return dists.argmax()

# def _get_largest(boxes, ref_box=None):
#     return np.array([volume(box) for box in boxes]).argmax()

# def _get_smallest(boxes, ref_box=None):
#     return np.array([volume(box) for box in boxes]).argmin()

def _closest(ref_centroid, centroids):
    dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
    if len(dists)==0:
        return -2
    return dists.argmin()

def _farthest(ref_centroid, centroids):
    dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
    return dists.argmax()

def _is_above(ref_center, center):
    y_diff = ref_center[1] - center[1]
    is_relation = y_diff > 0.03
    return is_relation

def _is_below(ref_center, center):
    y_diff = center[1] - ref_center[1]
    is_relation = y_diff > 0.03
    return is_relation
    
def _is_similar_height(ref_center, center):
    y_diff_abs = np.abs(ref_center[1] - center[1])
    is_relation = y_diff_abs <= 0.06
    return is_relation

def _is_supported_by(ref_centroid, centroids, ground_plane_h=None):

    # first check supported by floor
    floor_dist = ref_centroid[1] - ground_plane_h
    if floor_dist<0.1:
        return -1 # floor

    # must be below 
    obj_below = (centroids[:,1] - ref_centroid[1]) < 0.03
    dists = np.sqrt(np.sum((centroids[obj_below] - np.expand_dims(ref_centroid, axis=0))**2, axis=1))

    if len(dists)==0:
        return -2

    argmin_dist = dists.argmin()
    argmin_ = np.arange(centroids.shape[0])[obj_below][argmin_dist]

    return argmin_

def _is_next_to(ref_center, center):
    dist = np.sqrt(np.sum((center - ref_center)**2))
    is_relation = dist < 1.2
    return is_relation

relations_executors_pairs = {
                # 'above': relations_util._is_above,
                # 'below': relations_util._is_below,
                'next-to': _is_next_to,
                'supported-by': _is_supported_by,
                # 'similar-height-to': relations_util._is_similar_height,
                # 'farthest-to': relations_util._farthest,
                'closest-to': _closest,
            }

receptacles = ['Cabinet', 'CounterTop', 'Sink', 'TowelHolder',
                'GarbageCan', 
                'SinkBasin', 'Bed', 
                'Drawer', 'SideTable', 'Chair', 'Desk', 'Dresser',  
                'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'ShelvingUnit', 
                'Shelf', 'StoveBurner', 'Microwave', 'CoffeeMachine', 'Fridge', 
                'Toaster', 'DiningTable',  
                'LaundryHamper', 'Stool', 'CoffeeTable', 'Bathtub', 'Footstool', 'BathtubBasin', 
                'TVStand', 'Safe']