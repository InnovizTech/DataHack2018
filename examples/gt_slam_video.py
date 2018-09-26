# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.

import pandas as pd
import os
import numpy as np
from utilities.math_utils import RotationTranslationData
from visualizations.vis import pcshow
from utilities import data_utils


if __name__ == '__main__':
    base_dir = os.path.dirname(os.getcwd())
    video_dir = os.path.join(base_dir, 'data_examples', 'test_video')
    agg_point_cloud_list = []
    max_frames_to_keep = 10
    min_idx = 0
    decimate = 1
    max_dist = 100
    for idx in data_utils.enumerate_frames(video_dir):
        if idx < min_idx or idx % decimate != 0:
            continue
        pc_file = data_utils.frame_to_filename(video_dir, idx, 'pointcloud')
        pc, ego, label = data_utils.read_all_data(video_dir, idx)
        ego_rt = RotationTranslationData(vecs=(ego[:3], ego[3:]))
        ego_pc = ego_rt.apply_transform(pc[:, :3])
        ego_pc = np.concatenate((ego_pc, pc[:, 3:4]), -1)

        labeled_pc = np.concatenate((ego_pc, label), -1)
        agg_point_cloud_list.append(labeled_pc)
        if len(agg_point_cloud_list) > max_frames_to_keep:
            agg_point_cloud_list = agg_point_cloud_list[1:]
        agg_point_cloud = np.concatenate(agg_point_cloud_list, 0)
        pc2disp = ego_rt.inverse().apply_transform(agg_point_cloud[:, :3])
        pc2disp = np.concatenate((pc2disp, agg_point_cloud[:, 3:]), -1)
        pc2disp = pc2disp[np.linalg.norm(pc2disp[:, :3], axis=1) < max_dist]
        pcshow(pc2disp, on_screen_text=pc_file, max_points=32000 * max_frames_to_keep)
