# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.

import os.path as osp
import glob
import pandas as pd
import numpy as np
import re


def find_sorted_files(main_dir, file_type):
    # find all point cloud files
    path_re = osp.join(main_dir, '[0-9]*_' + file_type + '.csv')
    return sorted(glob.glob(path_re))


def read_all_data(folder, index, mounting_vector=(0.6, 0., 1.3)):
    """
    Reads all (available) data files and "fixes" Lidar coordinate system origin.
    :param folder:
    :param index:
    :return:
    """
    pc = read_data(folder, index, 'pointcloud')
    pc[:, :3] += np.array(mounting_vector)
    try:
        ego_motion = read_data(folder, index, 'egomotion')
    except FileNotFoundError:
        ego_motion = None

    try:
        labels = read_data(folder, index, 'labels')
    except FileNotFoundError:
        labels = None

    return pc, ego_motion, labels



def read_data(folder, index, data_type):
    """
    Reads data from files.
    :param str folder: folder with required video
    :param int index: frame index
    :param str data_type: data to be read. Options are: 'pointcloud', 'egomotion', 'labels'
    :return:
    """
    file_name = frame_to_filename(folder, index, data_type)
    cm_to_m_factor = 0.01
    if data_type == 'pointcloud':
        data = pd.read_csv(file_name, delimiter=',', header=None).values
        return data * cm_to_m_factor
    if data_type == 'egomotion':
        data = pd.read_csv(file_name, delimiter=',', header=None).values.ravel()
        return data
    if data_type == 'labels':
        return pd.read_csv(file_name, delimiter=',', dtype=int, header=None).values


def enumerate_frames(folder):
    for file in find_sorted_files(folder, 'pointcloud'):
        yield filename_to_frame(file)


def data_exists(folder, frame, data_type):
    try:
        read_data(folder, frame, data_type)
        return True
    except FileNotFoundError:
        return False


def extract_frame_indices(folder):
    """
    returns a sorted list of all frames indices in a folder
    :param folder:
    :return: frame_indices
    """
    return list(enumerate_frames(folder))


def count_frames(folder):
    """
    Counts the number of frames in a folder
    :param folder:
    :return:
    """
    point_cloud_files = find_sorted_files(folder, 'pointcloud')
    return len(point_cloud_files)


def filename_to_frame(file_name):
    return int(osp.basename(file_name).split('_')[0])


def frame_to_filename(folder, index, data_type):
    valid_input = {'pointcloud', 'egomotion', 'labels'}
    if data_type not in valid_input:
        raise ValueError("Input data type has to be from {}".format(valid_input))
    return osp.join(folder, str(index).zfill(7) + '_' + data_type + '.csv')
