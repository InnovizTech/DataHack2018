# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import shapely.geometry
import shapely.affinity
from shapely.prepared import prep
from descartes import PolygonPatch


class RotationTranslationData(object):
    """
    Abstraction class that represents rigid body transformation and can accept rotations as quaternions, rotation angles
    around axes
    """
    def __init__(self, mat=None, vecs=None, rt=None):
        """

        :param mat: 4x4 transformation matrix. translation is at the last column.
        :param vecs: tuple of two vectors (rotation, translation). If rotation has 3 elements it is assumed to be
            rotation angles around the axes (x, y, z), if it has 4 elements it is assumed to be quaternion
        :param rt: tuple of a 3x3 matrix and a 3 element vector of rotation and translation respectively
        """
        self._rotation_angles = None
        self._translation = None
        self._axis_angle = None
        self._q = None
        self._mat = None
        assert sum((mat is not None, vecs is not None, rt is not None)) == 1

        # received 4x4 matrix as input
        if mat is not None:
            assert (mat.shape == (4, 4))
            self._mat = mat
            self._rotation = self._mat[:3, :3]
            self._translation = self._mat[:3, 3]
        # received rotation vector and translation vector
        elif vecs is not None:
            assert all(isinstance(vec, np.ndarray) for vec in vecs)
            assert len(vecs) == 2 and vecs[1].shape == (3,)
            if vecs[0].shape == (3,):
                # rotation angles
                self._rotation_angles = vecs[0]
                self._rotation = rotation_angles_to_rotation(vecs[0])
                self._translation = vecs[1]
            else:
                # quaternion
                assert vecs[0].shape == (4,)
                self._q = vecs[0]
                self._rotation = quaternion_to_rotation(vecs[0])
                self._translation = vecs[1]
        # received rotation matrix and translation vector
        else:
            assert len(rt) == 2 and all(isinstance(i, np.ndarray) for i in rt)
            assert rt[0].shape == (3, 3) and rt[1].shape == (3,)
            self._translation = rt[1]
            self._rotation = rt[0]

    @property
    def to_matrix(self):
        return rotation_translation_to_4d(self.rotation, self.translation)

    @property
    def rotation_angles(self):
        if self._rotation_angles is None:
            self._rotation_angles = extract_rotation(self.rotation)
        return self._rotation_angles

    @property
    def axis_angle(self):
        if self._axis_angle is None:
            self._axis_angle = extract_axis_angle(self.rotation)
        return self._axis_angle

    @property
    def quaternion(self):
        if self._q is None:
            self._q = extract_quaternion(self.rotation)
        return self._q

    @property
    def to_rt(self):
        return self.rotation_angles, self.translation

    @property
    def to_axis_angle_translation(self):
        return self.axis_angle, self.translation

    @property
    def to_quaternion_translation(self):
        return self.quaternion, self.translation

    @property
    def translation(self):
        if self._translation is None:
            self._translation = extract_translation(self._mat)
        return self._translation

    @property
    def rotation(self):
        return self._rotation

    def apply_transform(self, points):
        assert points.ndim == 2 or points.ndim == 1
        assert points.shape[-1] == 3
        return np.matmul(points, self._rotation.T) + self._translation[None, :]

    def apply_transform_complete_pc(self, points):
        assert points.ndim == 2 or points.ndim == 1
        assert points.shape[-1] == 3 or points.shape[-1] == 4 or points.shape[-1] == 5
        pc = points[:, :3]
        trans_pc = np.matmul(pc, self._rotation.T) + self._translation[None, :]

        return np.concatenate((trans_pc, points[:, 3:]), axis=-1)

    def inverse(self):
        '''
        :return: The inverse transformation
        '''
        _R = self.rotation.T
        _t = -np.dot(self.rotation.T, self.translation)
        return RotationTranslationData(rt=(_R, _t))

    def compose(self, transform):
        """
        Compose this above another transform. Assumes all matrices multiply vectors from the left (i.e. A*x).
        :param transform: transform to compose
        :return: composed transform _matrix * transform
        """
        assert isinstance(transform, RotationTranslationData)
        R_new = self.rotation.dot(transform.rotation)
        t_new = self.rotation.dot(transform.translation) + self.translation
        return RotationTranslationData(rt=(R_new, t_new))

    @classmethod
    def from_axis_angle_translation(cls, axis_angle, translation):
        assert isinstance(axis_angle, np.ndarray) and axis_angle.shape == (4,)
        assert isinstance(translation, np.ndarray) and translation.shape == (3,)
        R = axis_angle_to_rotation(axis_angle[1:], axis_angle[0])
        return RotationTranslationData(rt=(R, translation))

    @classmethod
    def align_plane_to_ground(cls, u, d):
        """
        Generates a rotation translation object that rotates such that u coincides with (0, 0, -1) and d=0
        :param vector1: the vector to be rotated
        :param vector2: the vector to be rotated to
        :return: RotationTranslationData
        """
        u = u.ravel()
        c = np.sqrt(1-u[0]**2)
        cinv = 1 / c

        R = np.array([[c, -u[0]*u[1]*cinv, -u[0]*u[2]*cinv],
                      [0, -u[2] * cinv, u[1] * cinv],
                      [-u[0], -u[1], -u[2]]])
        return RotationTranslationData(rt=(R, np.array([0., 0., d])))

    @staticmethod
    def identity():
        return RotationTranslationData(rt=(np.eye(3), np.zeros(3)))

    def __str__(self):
        return "<RotationTranslationData: Rotation=({:.2f}, {:.2f}, {:.2f}), Translation=({:.2f}, {:.2f}, {:.2f})>"\
            .format(self.rotation_angles[0], self.rotation_angles[1], self.rotation_angles[2],
                    self.translation[0], self.translation[1], self.translation[2])


def euler_matrix(yaw, pitch, roll):
    # angles in radians
    # negating pitch for easier computation
    pi=np.pi+1e-7
    assert -pi <= yaw <= pi and -np.pi <= roll <= np.pi and -np.pi/2 <= pitch <= np.pi/2, \
        "Erroneous yaw, pitch, roll={},{},{}".format(yaw, pitch, roll)
    rotX = np.eye(3)
    rotY = np.eye(3)
    rotZ = np.eye(3)
    rotX[1:, 1:] = rot_mat_2d(roll)
    rotY[::2, ::2] = rot_mat_2d(pitch)
    rotZ[:2, :2] = rot_mat_2d(yaw)

    return rotZ.dot(rotY.dot(rotX))


def yaw(points):
    # calculate yaw of points given in nx3 format. yaw in [-pi, pi]
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3
    return np.arctan2(points[:, 1], points[:, 0])


def pitch(points):
    # calculate pitch of points given in nx3 format. pitch in [-pi/2, pi/2]
    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3
    return np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1))


def rot_mat_2d(angle):
    # angle in radians
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def extract_rotation(mat):
    # According to element [1,1] in table in https://en.wikipedia.org/wiki/Euler_angles (X1Y2Z3)
    # returns (roll, pitch, yaw) or (X angle, Y angle, Z angle)
    return np.array([np.arctan2(-mat[1, 2], mat[2, 2]), np.arcsin(mat[0, 2]), np.arctan2(-mat[0, 1], mat[0, 0])])


def extract_translation(mat):
    return np.array((mat[0, 3], mat[1, 3], mat[2, 3]))


def rotation_translation_to_4d(rotation, translation):
    """ returns 4x4 rotation translation matrix given a 3x3 rotation matrix and 3 translation vector"""
    return np.vstack((np.hstack((rotation,translation.reshape((-1, 1)))), np.array((0, 0, 0, 1))))


def pose_from_rt(rotation, translation):
    # rotation is received as roll, pitch, yaw
    # Convert to rotation translation. Roll and yaw angles are in [-pi, pi] while pitch is [-pi/2, pi/2]
    return rotation_translation_to_4d(rotation_angles_to_rotation(rotation), translation)


def pose_from_axis_angle(axis, angle, translation):
    return rotation_translation_to_4d(axis_angle_to_rotation(axis, angle), translation)


def pose_from_quaternion(q, translation):
    return rotation_translation_to_4d(quaternion_to_rotation(q), translation)


def rotation_angles_to_rotation(rotation):
    return euler_matrix(rotation[2], rotation[1], rotation[0])


def axis_angle_to_rotation(axis, angle):
    # converts axis angle notation to a rotation matrix. angle is assumed in radians and in [0, 2pi],
    # axis should be normalized.
    # formula from https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis-angle
    assert isinstance(axis, np.ndarray) and axis.shape == (3,)
    assert np.abs(np.linalg.norm(axis) - 1.) < 1e-6
    assert 0 <= angle <= np.pi * 2
    rotation_matrix = np.cos(angle) * np.eye(3) + np.sin(angle) * cross_product_matrix(axis) + \
                      (1 - np.cos(angle)) * np.tensordot(axis, axis, axes=0)
    return rotation_matrix


def cross_product_matrix(vector):
    assert isinstance(vector, np.ndarray) and vector.shape == (3,)
    matrix = np.array([[0, -vector[2], vector[1]],
                       [vector[2], 0, -vector[0]],
                       [-vector[1], vector[0], 0]])
    return matrix


def extract_axis_angle(rot_mat):
    # Convert from rotation matrix to axis angle. This conversion is good for angles in [0, pi], angles in [-pi, 0] will
    # be mapped to [pi, 0] (pi downto 0) with the negative axis. To handle this issue we can use quaternions.
    assert isinstance(rot_mat, np.ndarray) and rot_mat.shape == (3, 3,)
    u = np.array([rot_mat[2, 1] - rot_mat[1, 2],
                  rot_mat[0, 2] - rot_mat[2, 0],
                  rot_mat[1, 0] - rot_mat[0, 1]])
    angle = np.arccos(np.trace(rot_mat[:3, :3]) / 2 - 0.5)
    if np.linalg.norm(u) == 0.:
        return np.array([0., 0., 0., 1.])
    else:
        u = u / np.linalg.norm(u)
        return np.array([angle, u[0], u[1], u[2]])


def extract_quaternion(R):
    d = np.diagonal(R)
    t = np.sum(d)
    if t + 1 < 0.25:
        symmetric_mat = R + R.T
        asymmetric_mat = R - R.T
        symmetric_diag = np.diagonal(symmetric_mat)
        i_max = np.argmax(symmetric_diag)
        q = np.empty(4)
        if i_max == 0:
            q[1] = np.sqrt(symmetric_diag[0] - t + 1) / 2
            normalizer = 1 / q[1]
            q[2] = symmetric_mat[1, 0] / 4 * normalizer
            q[3] = symmetric_mat[2, 0] / 4 * normalizer
            q[0] = asymmetric_mat[2, 1] / 4 * normalizer
        elif i_max == 1:
            q[2] = np.sqrt(symmetric_diag[1] - t + 1) / 2
            normalizer = 1 / q[2]
            q[1] = symmetric_mat[1, 0] / 4 * normalizer
            q[3] = symmetric_mat[2, 1] / 4 * normalizer
            q[0] = asymmetric_mat[0, 2] / 4 * normalizer
        elif i_max == 2:
            q[3] = np.sqrt(symmetric_diag[2] - t + 1) / 2
            normalizer = 1 / q[3]
            q[1] = symmetric_mat[2, 0] / 4 * normalizer
            q[2] = symmetric_mat[1, 2] / 4 * normalizer
            q[0] = asymmetric_mat[1, 0] / 4 * normalizer
    else:
        r = np.sqrt(1+t)
        s = 0.5 / r
        q = np.array([0.5*r, (R[2, 1] - R[1, 2])*s, (R[0, 2] - R[2, 0])*s, (R[1, 0] - R[0, 1])*s])

    return q


def quaternion_to_rotation(q):
    """
    Conversion from quaternion vector (w,x,y,z) to a rotation matrix.
    Based on https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    """
    w, x, y, z = tuple(q)
    n = np.dot(q, q)
    s = 0. if n == 0. else 2./n
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z

    R = np.array([[1-yy-zz, xy-wz, xz+wy],
                  [xy+wz, 1-xx-zz, yz-wx],
                  [xz-wy, yz+wx, 1-xx-yy]])
    return R


class Box(object):
    def __init__(self, x, y, wx, wy, rotation, h):
        self._center = np.array([x, y])
        self._width = np.array([wx, wy])
        self._rotation = rotation
        self._height = h

        c = shapely.geometry.box(-wx / 2.0, -wy / 2.0, wx / 2.0, wy / 2.0)
        rc = shapely.affinity.rotate(c, -rotation, use_radians=True)
        trc = shapely.affinity.translate(rc, x, y)

        self._contour = trc

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def rotation(self):
        return self._rotation

    @property
    def contour(self):
        return self._contour

    def intersects(self, other):
        assert isinstance(other, Box)
        return self.contour.intersects(other.contour)

    def _intersection(self, other):
        return self.contour.intersection(other.contour)

    def intersection_many(self, others):
        if len(others) > 1:
            prepped_contour = prep(self.contour)
        else:
            prepped_contour = self.contour
        intersections = [0.] * len(others)
        for idx, other in enumerate(others):
            if prepped_contour.intersects(other.contour):
                intersections[idx] = self._intersection(other).area
        return intersections

    def union(self, other):
        return self.contour.union(other.contour)

    def iou(self, other):
        intersection_area = self._intersection(other).area
        return intersection_area / (self.area + other.area - intersection_area + 1e-9)

    def iou_many(self, others):
        if len(others) > 1:
            prepped_contour = prep(self.contour)
        else:
            prepped_contour = self.contour
        ious = [0.] * len(others)
        for idx, other in enumerate(others):
            if prepped_contour.intersects(other.contour):
                ious[idx] = self.iou(other)
        return ious

    @classmethod
    def boxes_iou(cls, box1, box2):
        return box1.iou(box2)

    @property
    def area(self):
        return self.contour.area

    def draw(self, ax, color, line_width=1, fillcolor=None, name=None, arrow=True, alpha=0.2, scale=50):
        ax.add_patch(PolygonPatch(self.contour, alpha=alpha, fc=fillcolor, ec=color, linewidth=line_width))

        vertices = np.array(self.contour.exterior.coords)[1:]

        if arrow:
            arrow_center = np.mean(vertices, axis=0)
            arrow_direction = (vertices[2] - vertices[1]) / 1.5
            arrow_tail = arrow_center - arrow_direction / 2
            arrow_head = arrow_center + arrow_direction / 2
            style = plt_patches.ArrowStyle.Simple(head_length=.4, head_width=.6, tail_width=.1)
            x = np.array(ax.axis())
            scale_factor = np.sqrt(np.prod(np.abs(x[::2] - x[1::2])) / (60 * 60))
            arrow_patch = plt_patches.FancyArrowPatch(posA=arrow_tail, posB=arrow_head, arrowstyle=style,
                                                      color='w', mutation_scale= scale / scale_factor, alpha=0.4)
            ax.add_patch(arrow_patch)
        elif name is None:
            name = 'front'

        if name is not None:
            text_location = np.mean(vertices[[0, -1]], axis=0)
            ax.text(text_location[0], text_location[1], name, ha='center', va='top', color='w')

    @classmethod
    def from_numpy(cls, numpy_arr):
        # assumes input as (x,y,wx,wy,h,angle)
        assert isinstance(numpy_arr, np.ndarray)
        assert (numpy_arr.ndim == 1 and numpy_arr.size == 6) or (numpy_arr.ndim == 2 and numpy_arr.shape[1] == 6)
        if numpy_arr.ndim == 1:
            return Box(numpy_arr[0], numpy_arr[1], numpy_arr[2], numpy_arr[3], numpy_arr[5], numpy_arr[4])
        else:
            return [Box(numpy_arr[i, 0], numpy_arr[i, 1], numpy_arr[i, 2],
                        numpy_arr[i, 3], numpy_arr[i, 5], numpy_arr[i, 4]) for i in range(numpy_arr.shape[0])]

    @classmethod
    def from_xyxy(cls, box):
        center = 0.5*(box[:2] + box[2:])
        width = (box[2:] - box[:2])
        return Box(center[0], center[1], width[0], width[1], 0., 0.)


def draw_point(point, axes=None, color='k'):
    circle = plt.Circle((point[0], point[1]), 0.005, color='k')
    if axes is None:
        axes = plt
    axes.add_artist(circle)


def box2dtobox3d(boxes2d, z_translation=0.0, z_size=0.0, z_angle=0.0):
    """
    tranforms 2d boxes to 3d boxes
    :param boxes2d: np array shaped N,4. box = [x1,y1,x2,xy] (1-bottom left, 2 upper right)
    :return: boxes3d np array shaped N,7. box = [t1,t2,t3,s1,s2,s3,z_angle]
    """
    ctr_x = np.mean(boxes2d[:, [0, 2]], axis=-1, keepdims=True)
    ctr_y = np.mean(boxes2d[:, [1, 3]], axis=-1, keepdims=True)
    ctr_z = np.full([boxes2d.shape[0], 1], z_translation)
    ctr = np.concatenate((ctr_x, ctr_y, ctr_z), -1)

    size_x = boxes2d[:, 2:3] - boxes2d[:, 0:1]
    size_y = boxes2d[:, 3:4] - boxes2d[:, 1:2]
    size_z = np.full([boxes2d.shape[0], 1], z_size)
    size = np.concatenate((size_x, size_y, size_z), -1)

    z_angle = np.full([boxes2d.shape[0], 1], z_angle)

    return np.concatenate((ctr, size, z_angle), -1)
