from shapely.geometry import Polygon
import numpy as np

def lwh_to_box(l, w, h):
    box = np.array([
        [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
    ])
    return box


def intersect_bbox_with_yaw(box_a, box_b):
    """
    Obtain box_a and box_b from Bbox3d.get_bbox
    A simplified calculation of 3d bounding box intersection.
    It is assumed that the bounding box is only rotated
    around Z axis (yaw) from an axis-aligned box.
    :param box_a, box_b: obstacle bounding boxes for comparison
    :return: intersection volume (float)
    """
    # height (Z) overlap
    min_h_a = np.min(box_a[2])
    max_h_a = np.max(box_a[2])
    min_h_b = np.min(box_b[2])
    max_h_b = np.max(box_b[2])
    max_of_min = np.max([min_h_a, min_h_b])
    min_of_max = np.min([max_h_a, max_h_b])
    z_intersection = np.max([0, min_of_max - max_of_min])
    if z_intersection == 0:
        return 0.

    # oriented XY overlap
    xy_poly_a = Polygon(zip(*box_a[0:2, 0:4]))
    xy_poly_b = Polygon(zip(*box_b[0:2, 0:4]))
    xy_intersection = xy_poly_a.intersection(xy_poly_b).area
    if xy_intersection == 0:
        return 0.

    return z_intersection * xy_intersection

def iou(vol_a, vol_b, vol_intersect):
    '''
    Obtain vol_a, vol_b from Bbox3d.get_vol_box
    Obtain vol_intersect from intersect_bbox_with_yaw
    '''
    union = vol_a + vol_b - vol_intersect
    return vol_intersect / union if union else 0.


class Bbox3d(object):

    def __init__(self, object_type, height, width, length, x, y, z, rot_y):
        '''object_type is a the string of object class
        '''
        self.object_type = object_type
        self.h = height
        self.w = width
        self.l = length
        self.position = [x, z, y]
        self.yaw = rot_y
        self._oriented_bbox = None  # for caching

    def get_bbox(self):
        if self._oriented_bbox is None:
            bbox = lwh_to_box(self.l, self.w, self.h)
            # calc 3D bound box in capture vehicle oriented coordinates
            rot_mat = np.array([
                [np.cos(self.yaw), -np.sin(self.yaw), 0.0],
                [np.sin(self.yaw), np.cos(self.yaw), 0.0],
                [0.0, 0.0, 1.0]])
            self._oriented_bbox = np.dot(rot_mat, bbox) + np.tile(self.position, (8, 1)).T
        return self._oriented_bbox

    def get_vol_box(self):
        return self.h * self.w * self.l

    def get_vol(self):
        return self.get_vol_box()

    def intersection_metric(self, other, metric_fn=iou):
        intersection_vol = intersect_bbox_with_yaw(self.get_bbox(), other.get_bbox())
        metric_val = metric_fn(self.get_vol_box(), other.get_vol_box(), intersection_vol)
        return metric_val, intersection_vol

    def __repr__(self):
        return str(self.object_type) + ' ' + str(self.h) + ' ' + str(self.w) + ' ' + str(self.l) + ' ' + str(self.position) + ' ' + str(self.yaw)