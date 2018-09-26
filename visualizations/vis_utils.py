# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.

import numpy as np
import os.path as osp

from direct.showbase.DirectObject import DirectObject
from direct.showbase.ShowBase import ShowBase

from panda3d.core import TextNode, Shader, TransparencyAttrib, NodePath, BoundingBox, LineSegs, deg2Rad
from panda3d.core import GeomVertexFormat, Geom, GeomVertexData, GeomVertexWriter, GeomNode, GeomTriangles, GeomLines
from panda3d.core import Filename
from direct.gui.DirectGui import OnscreenText

try:
    from .load_point_cloud import load_point_cloud, set_texture
except ImportError as ex:
    def load_point_cloud(pos_writer, color_writer, point_cloud, color, point_cloud_size, prev_point_cloud_size):
        pos_writer.set_row(0)
        color_writer.set_row(0)
        color[color > 1] = 1
        total_points = point_cloud.shape[0]
        for pnt_idx in range(min(point_cloud_size, max(total_points, prev_point_cloud_size))):
            if pnt_idx < total_points:
                for i in range(6):
                    pos_writer.setData3f(float(point_cloud[pnt_idx, 0]), float(point_cloud[pnt_idx, 1]),
                                         float(point_cloud[pnt_idx, 2]))
                    if color is not None:
                        color_writer.setData4(color[pnt_idx, 0], color[pnt_idx, 1], color[pnt_idx, 2],
                                              color[pnt_idx, 3])
            else:
                for i in range(6):
                    color_writer.setData4(0., 0., 0., 0.)


    def set_texture(tex_writer, point_cloud_size):
        for pnt_idx in range(point_cloud_size):
            tex_writer.setData2f(-0.5, +0.5)
            tex_writer.setData2f(-0.5, -0.5)
            tex_writer.setData2f(+0.5, -0.5)
            tex_writer.setData2f(+0.5, +0.5)
            tex_writer.setData2f(-0.5, +0.5)
            tex_writer.setData2f(+0.5, -0.5)


def rot_mat_2d(angle):
    # angle in radians
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


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
    rotY[::2, ::2] = rot_mat_2d(-pitch)
    rotZ[:2, :2] = rot_mat_2d(yaw)

    return rotX.dot(rotY.dot(rotZ))


def makeArc(angleDegrees = 360, numSteps = 16, scale=2,color=(1,1,1,1)):
    ls = LineSegs()
    ls.setColor(color)
    angleRadians = deg2Rad(angleDegrees)

    for i in range(numSteps + 1):
        a = angleRadians * i / numSteps
        y = np.sin(a)*scale
        x = np.cos(a)*scale

        ls.drawTo(y, x, 0)
    if angleDegrees != 360:
        for i in range(numSteps + 1):
            a = -angleRadians * i / numSteps
            y = np.sin(a)*scale
            x = np.cos(a)*scale

            ls.drawTo(y, x, 0)

    node = ls.create()
    return NodePath(node)


class GroundGen(object):
    def __init__(self, max_r):
        self.max_r = max_r
        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('point', format, Geom.UHDynamic)
        self._pos_writer = GeomVertexWriter(vdata, 'vertex')
        self._color_writer = GeomVertexWriter(vdata, 'color')

        line_num = 60
        vdata.setNumRows(line_num)

        angles = np.linspace(0, np.pi * 2 - np.pi * 2 / line_num , line_num)

        other_rgba = (0., 0., 0.3, 0.1)
        other2_rgba = (0.1, 0.1, 0.4, 0.4)
        axis_rgba = (0.2, 0.2, 0.5, 1.0)
        max_r = 250
        for indx, angle in enumerate(angles):
            if indx % 5 == 0:
                rgba = axis_rgba
            else:
                rgba = other_rgba
            self._pos_writer.addData3d(0, 0, 0.)
            self._color_writer.addData4f(rgba[0], rgba[1], rgba[2], rgba[3])
            self._pos_writer.addData3d(max_r * np.sin(angle), max_r * np.cos(angle), 0.)
            self._color_writer.addData4f(rgba[0], rgba[1], rgba[2], rgba[3])

        grnd_prmtv = GeomLines(Geom.UHStatic)
        grnd_prmtv.addConsecutiveVertices(0, 2 * line_num)
        grnd_prmtv.closePrimitive()
        ground_geom = Geom(vdata)
        ground_geom.addPrimitive(grnd_prmtv)
        snode = GeomNode('ground_lines')
        snode.addGeom(ground_geom)

        self.points_node = base.render.attachNewNode(snode)
        self.points_node.setTwoSided(True)

        for rad in range(int(max_r)):
            color = axis_rgba
            pp = makeArc(angleDegrees=360, numSteps=160, scale=rad, color=color)
            tn = TextNode('dd')
            tn.setText(str(rad))
            tn.setTextScale(0.2)
            tn.setTextColor(color)
            text_geom = GeomNode('text')
            text_geom.addChild(tn)
            tp = NodePath(text_geom)
            tp.setPos((0, rad-0.2, 0))
            tp.setHpr((0, -90, 0))
            tp.reparentTo(self.points_node)
            pp.reparentTo(self.points_node)


class PointCloudVertexBuffer(object):
    def __init__(self, point_cloud_size):
        format = GeomVertexFormat.getV3c4t2()
        vdata = GeomVertexData('point', format, Geom.UHDynamic)
        self._pos_writer = GeomVertexWriter(vdata, 'vertex')
        self._color_writer = GeomVertexWriter(vdata, 'color')
        self._tex_writer = GeomVertexWriter(vdata, 'texcoord')
        self._point_cloud_size = point_cloud_size
        self._prev_point_cloud_size = 0

        assert point_cloud_size > 0
        vdata.setNumRows(point_cloud_size * 6)
        self._tex_writer.set_row(0)
        set_texture(self._tex_writer, point_cloud_size)
        pnts = GeomTriangles(Geom.UHStatic)
        pnts.addConsecutiveVertices(0, 3 * 2 * point_cloud_size)
        pnts.closePrimitive()
        points_geom = Geom(vdata)
        points_geom.addPrimitive(pnts)
        snode = GeomNode('points')
        snode.addGeom(points_geom)
        dir_name = osp.dirname(__file__)
        # print(osp.join(dir_name, 'pnts_vs.glsl'))
        vs_shader = osp.join(dir_name, 'pnts_vs.glsl')
        fs_shader = osp.join(dir_name, 'pnts_fs.glsl')
        myShader = Shader.load(Shader.SL_GLSL, vertex=Filename.fromOsSpecific(vs_shader).getFullpath(),
                               fragment=Filename.fromOsSpecific(fs_shader).getFullpath())

        assert myShader is not None
        self.points_node = base.render.attachNewNode(snode)
        self.points_node.setPos(0., 0., 0.)
        self.points_node.set_shader(myShader)
        self.points_node.set_shader_input("view_size", (base.win.getXSize(), base.win.getYSize()))
        self.points_node.node().setBounds(BoundingBox((-1000., -1000., -1000.), (1000., 1000., 1000.)))
        self.points_node.setTransparency(TransparencyAttrib.MAlpha)

    def assign_points(self, point_cloud, color):
        assert color.shape[1] == 4
        assert color.shape[0] == point_cloud.shape[0]
        load_point_cloud(self._pos_writer, self._color_writer, point_cloud.astype(np.float64), color.astype(np.float64),
                         self._point_cloud_size, self._prev_point_cloud_size)
        self._prev_point_cloud_size = point_cloud.shape[0]

    def clear_pc(self):
        self._color_writer.set_row(0)
        for i in range(self._point_cloud_size * 6):
            self._color_writer.setData4f(1., 1., 1., 0.)


class Cuboid(object):
    def __init__(self, size, translation, rotation, color, text):
        self._visible = False
        wy, wx, wz = size[0], size[1], size[2]
        format = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('cu_points', format, Geom.UHStatic)
        vdata.setNumRows(8)
        self._pos_writer = GeomVertexWriter(vdata, 'vertex')
        self._color_writer = GeomVertexWriter(vdata, 'color')
        self._pos_writer.set_row(0)
        self._color_writer.set_row(0)
        self._pos_writer.addData3f(-0.5 * wx, -0.5 * wy, 0.)
        self._pos_writer.addData3f(-0.5 * wx, -0.5 * wy, wz)
        self._pos_writer.addData3f(0.5 * wx, -0.5 * wy, wz)
        self._pos_writer.addData3f(0.5 * wx, -0.5 * wy, 0.)
        self._pos_writer.addData3f(-0.5 * wx, 0.5 * wy, 0.)
        self._pos_writer.addData3f(-0.5 * wx, 0.5 * wy, wz)
        self._pos_writer.addData3f(0.5 * wx, 0.5 * wy, wz)
        self._pos_writer.addData3f(0.5 * wx, 0.5 * wy, 0.)
        for i in range(8):
            self._color_writer.addData4f(color[0], color[1], color[2], color[3])

        lines = GeomLines(Geom.UHStatic)
        lines.addVertices(0, 1)
        lines.addVertices(1, 2)
        lines.addVertices(2, 3)
        lines.addVertices(3, 0)
        lines.addVertices(4, 5)
        lines.addVertices(5, 6)
        lines.addVertices(6, 7)
        lines.addVertices(7, 4)
        lines.addVertices(0, 4)
        lines.addVertices(1, 5)
        lines.addVertices(2, 6)
        lines.addVertices(3, 7)
        cuboid = Geom(vdata)
        cuboid.addPrimitive(lines)
        node = GeomNode('cuboid')
        node.addGeom(cuboid)
        self._node_path = NodePath(node)
        # self.title = OnscreenText(text=text, style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=.05,
        #                           parent=self._node_path, align=TextNode.ARight)

        self._txt_node = TextNode('id')
        self._txt_node.setText(text)
        self._txt_node.setTextScale(0.2)
        self._txt_node.setCardColor(0, 0, 1, 1)
        self._txt_node.setCardAsMargin(0, 0, 0, 0)
        self._txt_node.setCardDecal(True)
        self._txt_node.set_align(2)
        text_geom = GeomNode('text')
        text_geom.addChild(self._txt_node)
        self._txt_np = NodePath(text_geom)
        self._txt_np.reparentTo(self._node_path)

        self.show()
        self.update_values(size, translation, rotation, color, text)

    def update_values(self, size, translation, rotation, color, text):
        if self._visible:
            self.update_bb(self._node_path, self._pos_writer, self._color_writer, tuple(size),
                      tuple(translation), tuple(rotation), tuple(color), text, self._txt_node, self._txt_np)

    def hide(self):
        if self._visible:
            self._node_path.detachNode()
            self._visible = False

    def show(self):
        if not self._visible:
            self._node_path.reparentTo(base.render)
            self._node_path.setTwoSided(True)
            self._node_path.setTransparency(TransparencyAttrib.MAlpha)
            self._visible = True

    def update_bb(self, cuboid, pos_writer, color_writer, size, translation, rotation, color, text, text_node, text_np):
        wx, wy, wz = size
        pos_writer.setRow(0)
        color_writer.setRow(0)
        pos_writer.setData3f(-0.5 * wx, -0.5 * wy, 0.)
        pos_writer.setData3f(-0.5 * wx, -0.5 * wy, wz)
        pos_writer.setData3f(0.5 * wx, -0.5 * wy, wz)
        pos_writer.setData3f(0.5 * wx, -0.5 * wy, 0.)
        pos_writer.setData3f(-0.5 * wx, 0.5 * wy, 0.)
        pos_writer.setData3f(-0.5 * wx, 0.5 * wy, wz)
        pos_writer.setData3f(0.5 * wx, 0.5 * wy, wz)
        pos_writer.setData3f(0.5 * wx, 0.5 * wy, 0.)
        r, g, b, a = color
        for i in range(8):
            color_writer.setData4f(r, g, b, a)

        tx, ty, tz = translation
        ty = -ty
        rx, ry, rz = rotation
        rz = -rz
        cuboid.setPos(ty, tx, tz)
        cuboid.setHpr(-rz - 90., ry, rx)
        text_node.setText(text)
        text_np.setPos((0., 0., wz + 0.2))
        text_np.setHpr(rz + 90., 0, 0)


class Navigator3D(DirectObject):
    def __init__(self):
        DirectObject.__init__(self)
        self._prev_mouse_location = np.array([0., 0.,])
        self._prev_mouse_location_pen = None
        self._buttons_state = {}
        self._define_keymap()
        self._update_location_text()
        self.ground = GroundGen(150)
        taskMgr.add(self._update_camera, "updatecamera")

    def _register_keys(self, key_list):
        for key in key_list:
            self._register_key(key)

    def _register_key(self, key):
        assert isinstance(key, str)
        self._buttons_state[key] = False
        self.accept(key + '-up', self._key_up, [key])
        self.accept(key, self._key_down, [key])

    def _define_keymap(self):
        self._register_keys(['control', 'w', 's', 'a', 'd', 'mouse1', 'mouse3', 'alt'])
        self.accept('r', self._reset_view)
        self.accept('t', self._top_view)
        self.accept('c', self._top_view_centered)
        self.accept('wheel_up', lambda: self._zoom('+z'))
        self.accept('wheel_down', lambda: self._zoom('-z'))

    def _reset_view(self):
        base.camera.setPos(0, -10, 3)
        base.camera.setHpr(0, -9, 0)
        self._update_location_text()

    def _top_view(self):
        base.camera.setPos(0, 7, 30)
        base.camera.setHpr(0, -90, 0)
        self._update_location_text()

    def _top_view_centered(self):
        base.camera.setPos(0, 0, 60)
        base.camera.setHpr(0, -90, 0)
        self._update_location_text()

    def _update_location_text(self):
        pos = base.camera.getPos()
        hpr = base.camera.getHpr()
        if 'title' not in dir(self):
            self.title = OnscreenText(text="".format(tuple(pos), tuple(hpr)),
                                      style=1, fg=(1, 1, 1, 1), pos=(-0.1, 0.1), scale=.05,
                                      parent=base.a2dBottomRight, align=TextNode.ARight)
        self.title.setText("Position: ({:.2f}, {:.2f}, {:.2f})\n Rotation: ({:.2f}, {:.2f}, {:.2f})"
                           .format(pos[0], pos[1], pos[2], hpr[0], hpr[1], hpr[2]))

    def _pan_camera(self, movement):
        camera_pos = base.camera.getPos()
        hpr = base.camera.getHpr()
        rot_mat = lambda angle: np.array([[np.cos(angle), -np.sin(angle)],
                                          [np.sin(angle), np.cos(angle)]])
        movement_vec_xy = np.dot(rot_mat(hpr[0] / 180. * np.pi), np.array([movement[0], movement[1]]))
        base.camera.setPos(camera_pos[0] + movement_vec_xy[0],
                           camera_pos[1] + movement_vec_xy[1],
                           camera_pos[2] + movement[2])
        self._update_location_text()

    def _pan_camera_mouse(self, movement):
        camera_transform = np.array(base.camera.getNetTransform().getMat())
        pan_vec = np.array([-movement[0], 0, -movement[1], 1]).dot(camera_transform)
        base.camera.setPos(pan_vec[0],
                           pan_vec[1],
                           pan_vec[2])
        self._update_location_text()

    def _zoom(self, value):
        if type(value) is str:
            if value == '-z':
                value = -base.camera.getPos()[2]
            if value == '+z':
                value = base.camera.getPos()[2]
        scale = 0.3
        hpr = base.camera.getHpr()
        rotation_angles = np.array((-hpr[1], 0., -hpr[0])) / 180. * np.pi
        rotation = euler_matrix(rotation_angles[2], rotation_angles[1], rotation_angles[0])
        # inverse
        R = rotation.T
        offset = R.dot(np.array([[0., 1., 0.]]).T)

        new_pos = np.array(base.camera.getPos()) + offset.ravel() * scale * value
        new_pos[2] = max(new_pos[2], 0.)
        base.camera.setPos(new_pos[0], new_pos[1], new_pos[2])
        self._update_location_text()

    def _key_down(self, key):
        if key == 'mouse1':
            if base.mouseWatcherNode.hasMouse():
                self._buttons_state[key] = True
                self._prev_mouse_location = np.array(base.mouseWatcherNode.getMouse())
        else:
            self._buttons_state[key] = True

    def _key_up(self, key):
        self._buttons_state[key] = False
        if key == 'mouse1' and base.mouseWatcherNode.hasMouse():
            self._prev_mouse_location = np.array(base.mouseWatcherNode.getMouse())

    def _update_camera(self, task):
        scale = 0.3
        if 'w' in self._buttons_state and self._buttons_state['w']:
            if 'control' in self._buttons_state and self._buttons_state['control']:
                self._pan_camera((0., 0., scale))
            elif 'alt' in self._buttons_state and self._buttons_state['alt']:
                self._zoom(1.)
            else:
                self._pan_camera((0., scale, 0.))
        if 's' in self._buttons_state and self._buttons_state['s']:
            if 'control' in self._buttons_state and self._buttons_state['control']:
                self._pan_camera((0., 0., -scale))
            elif 'alt' in self._buttons_state and self._buttons_state['alt']:
                self._zoom(-1.)
            else:
                self._pan_camera((0., -scale, 0.))
        if 'a' in self._buttons_state and self._buttons_state['a']:
            self._pan_camera((-scale, 0., 0.))
        if 'd' in self._buttons_state and self._buttons_state['d']:
            self._pan_camera((scale, 0., 0.))

        if 'wheel_up' in self._buttons_state and self._buttons_state['wheel_up']:
            self._zoom(2^base.camera.getPos()[2])
        if 'wheel_down' in self._buttons_state and self._buttons_state['wheel_down']:
            self._zoom(-2^base.camera.getPos()[2])

        if 'mouse1' in self._buttons_state and self._buttons_state['mouse1'] and base.mouseWatcherNode.hasMouse():
            mouse_pos = np.array(base.mouseWatcherNode.getMouse())
            if np.linalg.norm(mouse_pos - self._prev_mouse_location) > 1e-2:
                prev_mouse_location = self._prev_mouse_location.copy()
                self._prev_mouse_location = mouse_pos
                diff = mouse_pos-prev_mouse_location
                self._rotate_camera(diff)

        if 'mouse3' in self._buttons_state and self._buttons_state['mouse3'] and base.mouseWatcherNode.hasMouse():
            mouse_pos = np.array(base.mouseWatcherNode.getMouse())
            if self._prev_mouse_location_pen is None or np.linalg.norm(mouse_pos - self._prev_mouse_location_pen) > 1e-2:
                if self._prev_mouse_location_pen is not None:
                    prev_mouse_location = self._prev_mouse_location_pen.copy()
                else:
                    prev_mouse_location = mouse_pos
                self._prev_mouse_location_pen = mouse_pos
                diff = (mouse_pos-prev_mouse_location) * np.clip(base.camera.getPos()[2], 5, 40)
                self._pan_camera_mouse((diff[0], diff[1]))
        else:
            self._prev_mouse_location_pen = None
        return task.cont

    def _rotate_camera(self, diff):
        rotation_scale = 30
        hpr = base.camera.getHpr()
        hpr[0] += diff[0] * rotation_scale
        hpr[1] -= diff[1] * rotation_scale
        base.camera.setHpr(hpr)
        self._update_location_text()


