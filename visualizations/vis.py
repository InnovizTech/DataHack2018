# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.

import numpy as np
import textwrap
import matplotlib.cm as cm
from multiprocessing import Process, Queue
from queue import Empty

from panda3d.core import ModifierButtons, TextNode
from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import OnscreenText

from visualizations.vis_utils import Navigator3D, PointCloudVertexBuffer, Cuboid
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import collections

pc_cmap = LinearSegmentedColormap.from_list('mycmap', ['red', 'yellow', 'limegreen'])
label_cmap = LinearSegmentedColormap.from_list('mycmap', ['white', 'magenta', 'blue'])
"""
This module allows visualization of point clouds. To use, simply use the pcshow() function.
"""


def pcshow(point_cloud=None, boxes=None, point_cloud_coloring='reflectivity_and_label', max_points=1000000, on_screen_text=None):
    """
    This is a convenience function that opens a PCDisplayer and displays point cloud. If used and
    another PCDisplayer is already opened, it will override it.
    """
    if PointCloudFrameViewer.main_display is None or not PointCloudFrameViewer.main_display.is_running:
        PointCloudFrameViewer.main_display = PointCloudFrameViewer(point_cloud, boxes, point_cloud_coloring, max_points, on_screen_text)
    else:
        PointCloudFrameViewer.main_display.display(point_cloud, boxes, point_cloud_coloring, on_screen_text)
    return PointCloudFrameViewer.main_display


class PointCloudFrameViewer(object):
    """
    This class is used to display a single point cloud frame that can be manipulated (e.g. navigated in 3d).
    In order to display stuff on it, use the display function, or use the paramters of the __init__ function.
    """
    main_display = None

    def __init__(self, point_cloud=None, boxes=None, point_cloud_coloring='reflectivity_and_label', max_points=17776, on_screen_text=None):
        self._queue = Queue(1)
        self._max_points = max(max_points, len(point_cloud))
        self._process = Process(target=self._gen_viewer, args=(self._queue, max_points))
        self._process.start()
        self.display(point_cloud, boxes, point_cloud_coloring, on_screen_text)
        self._cuboids = []

    def display(self, point_cloud, boxes, point_cloud_coloring='reflectivity_and_label', on_screen_text=None):
        if self._process.is_alive():
            d = {}
            if point_cloud is not None:
                d['point_cloud'] = point_cloud
                d['point_cloud_coloring'] = point_cloud_coloring
            if on_screen_text is not None:
                d['on_screen_text'] = on_screen_text
            if boxes is not None:
                d['boxes'] = boxes
            self._queue.put(d)
        else:
            raise Exception("point cloud display was closed")

    @staticmethod
    def _gen_viewer(queue, max_points=17776, massage_que=None):
        base = ShowBase()
        base.setBackgroundColor(0, 0, 0)
        base.disableMouse()
        base.camera.setPos(0, -50, 20)
        base.camera.setHpr(0, -22, 0)
        base.mouseWatcherNode.set_modifier_buttons(ModifierButtons())
        base.buttonThrowers[0].node().set_modifier_buttons(ModifierButtons())
        base.setFrameRateMeter(True)
        pc_viewer = _PointCloudFrameViewer(point_cloud_size=max_points, queue=queue)
        base.run()

    @property
    def is_running(self):
        return self._process.is_alive()


class PointCloudViewerApp(Navigator3D):
    def __init__(self, point_cloud_size=17776):
        Navigator3D.__init__(self)
        self._point_cloud = None
        self._point_cloud_coloring = None
        self._cuboids = []
        self.points_vb = PointCloudVertexBuffer(point_cloud_size)
        self._user_text = OnscreenText('', style=1, fg=(1, 1, 1, 1), scale=.04)
        self._user_text.setPos(-0.9, 0.9)

    def redraw_boxes(self):
        if self._boxes is not None:
            for c_index, box in enumerate(self._boxes):
                self._cuboids[c_index].show()

    def draw(self, point_cloud=None, point_cloud_coloring=None, on_screen_text=None, boxes=None):
        self._boxes = boxes
        if point_cloud is not None:
            pc_color = self.color_pc(point_cloud, point_cloud_coloring)
            self.draw_pc(point_cloud, pc_color)
        if on_screen_text is not None:
            self._user_text.setText(textwrap.fill(on_screen_text, 90))
        else:
            self._user_text.setText('')
        if boxes is not None:
            self.draw_cuboids(boxes)
            # self.redraw_boxes()

    def color_pc(self, pc, coloring='reflectivity_and_label', colormap='pc_cmap'):
        """
        Generate coloring for point cloud based on multiple options
        :param pc: point cloud
        :param coloring: Coloring option. Supported: 'reflectivity', np.array of point cloud size x 4 with points colors
        :return:
        """
        if colormap is 'pc_cmap':
            colormap = pc_cmap

        points = pc[:, :3]
        color = np.zeros((len(pc), 4))
        color[:, -1] = 1.

        if isinstance(coloring, np.ndarray) and coloring.dtype == np.int and coloring.shape == (points.shape[0],):
            cmap = ListedColormap(
                ['w', 'magenta', 'orange', 'mediumspringgreen', 'deepskyblue', 'pink', 'y', 'g', 'r', 'purple', ])
            coloring = np.mod(coloring, len(cmap.colors))
            c = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=len(cmap.colors)-1))
            color = c.to_rgba(coloring)
        elif isinstance(coloring, np.ndarray):
            if coloring.shape == (points.shape[0], 4):
                color = coloring
            if coloring.shape == (points.shape[0], ):
                c = cm.ScalarMappable(cmap=colormap)
                color = c.to_rgba(coloring, norm=False)
        elif isinstance(coloring, collections.Callable):
            colors = coloring(points)
            c = cm.ScalarMappable(cmap=colormap)
            color = c.to_rgba(colors)
        elif coloring == 'reflectivity':
            reflectivity = pc[:, 3]
            reflectivity[reflectivity > 1] = 1
            c = cm.ScalarMappable(cmap=colormap)
            color = c.to_rgba(reflectivity, norm=False)
            color[reflectivity < 0] = np.array([1.0, 1.0, 1.0, 1.0])
        elif coloring == 'reflectivity_and_label':
            # pc_colors
            reflectivity = pc[:, 3]
            reflectivity[reflectivity > 1] = 1
            c = cm.ScalarMappable(cmap=colormap)
            color = c.to_rgba(reflectivity, norm=False)
            if pc.shape[-1] == 5:
                labels = pc[:, 4]
                labels_valid = labels[labels > 0]
                c = cm.ScalarMappable(cmap=label_cmap)
                color_labels = c.to_rgba(labels_valid, norm=True)
                color[labels > 0] = color_labels
        else:
            color = np.ones((points.shape[0], 4))
            color[:, -1] = 1.
        return color

    def draw_pc(self, pc, color):
        points = pc[:, np.array([1, 0, 2])]
        self.points_vb.assign_points(points, color)

    def clear_point_cloud(self):
        self.points_vb.clear_pc()

    def draw_cuboids(self, boxes):
        for box_idx, box in enumerate(boxes):
            color = box['color'] if hasattr(box, 'color') else np.ones(4)
            size = box['size']
            translation = box['translation']
            rotation = box['rotation'] / np.pi * 180.
            try:
                text = box['text']
            except:
                text = ''

            if box_idx < len(self._cuboids):
                self._cuboids[box_idx].show()
                self._cuboids[box_idx].update_values(size, translation, rotation, color, text)
            else:
                self._cuboids.append(Cuboid(size, translation, rotation, color, text))
        for c in self._cuboids[len(boxes):]:
            c.hide()


class _PointCloudFrameViewer(PointCloudViewerApp):
    """
    Adds a queue on top of PointCloudViewerApp to get data for display asynchronously.
    """
    def __init__(self, point_cloud_size=17776, queue=None):
        PointCloudViewerApp.__init__(self, point_cloud_size)
        self._queue = queue
        taskMgr.add(self.dequeue, 'read_queue_task')

    def dequeue(self, task):
        if self._queue is None:
            return task.done
        try:
            disp_dict = self._queue.get(block=False, timeout=0.1)
        except Empty:
            return task.cont
        if disp_dict is not None:
            boxes = disp_dict['boxes'] if 'boxes' in disp_dict else None
            pc = disp_dict['point_cloud'] if 'point_cloud' in disp_dict else None
            on_screen_text = disp_dict['on_screen_text'] if 'on_screen_text' in disp_dict else None
            pc_coloring = 'reflectivity' if 'point_cloud_coloring' not in disp_dict else disp_dict['point_cloud_coloring']
            self.draw(pc, pc_coloring, on_screen_text, boxes)
        return task.cont


if __name__ == '__main__':
    file = './example.csv'
    labeled_pc = np.genfromtxt(file, delimiter=' ')
    labeled_pc[:, 3][labeled_pc[:, 4] == 1] = -1
    pcshow(labeled_pc[:, :4])
