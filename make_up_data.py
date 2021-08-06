#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   make_up_data.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/5/21 1:43 PM   StrikerCC    1.0         None
'''

# import lib
import copy

import numpy as np

import read_data
import vis
from transform import transform_rigid

dimension = 3


def sort_data_in_a_axis(points, axis=0):
    """
    reorder point sequence according to a specific axis
    reorder will mess up the natural sequence of points in array, can be used to test robustness of point cloud related algorithm

    :param points: points data: N points of 3D coordinates, shape expected to be (n, 3)
    :param axis: axis used to sort, expected to less than dimension of point data
    :return: a copy of points sorted already
    """
    assert isinstance(points, np.ndarray)
    assert points.shape == dimension, 'Expect points shape to be (N, 3), but get ' + str(points.shape)
    assert points.shape[1] < axis, 'Axis ' + str(axis) + ' is bigger than dimension ' + str(points.shape[1])
    index = np.argsort(points[:, axis], axis=0)
    index_x = index[:, 0]
    points = np.copy(points)
    points = points[index_x, :]
    return points


def add_Gaussian_noise(points, sigmas=1.0):
    """
    add Gaussian into points data, assume Gaussian noise distribution center at each point
    :param points: points data: N points of 3D coordinates, shape expected to be (n, 3)
    :param sigmas: standard deviation of Gaussian distribution
    :return:
    """
    assert isinstance(points, np.ndarray)
    assert points.shape[1] == dimension
    assert (isinstance(sigmas, list) or isinstance(sigmas, np.ndarray) and len(sigmas) == points.shape[1]) or isinstance(sigmas, float) or isinstance(sigmas, int)

    if not isinstance(sigmas, np.ndarray) and not isinstance(sigmas, list):
        sigmas = [sigmas, sigmas, sigmas]

    points = np.copy(points)
    points += np.random.normal(loc=[0.0 for _ in range(points.shape[-1])],
                               scale=sigmas,
                               size=points.shape)
    return points


def add_walls(points):
    """
    add a wall behind the object
    :param points:
    :return:
    """
    offset_factor = 2
    height_factor = 1.1
    width_factor = 2

    center = np.mean(points, axis=0)
    points_normalized = points - center
    offset = offset_factor * np.max(np.linalg.norm(points_normalized, axis=-1))
    height = height_factor * np.max(points_normalized[:, 2])
    width = width_factor * np.max(points_normalized[:, 0])


def add_floor(points):
    """
    add a floor under the object
    :param points:
    :return:
    """
    width_factor = 2

    center = np.mean(points, axis=0)
    points_normalized = points - center

    width = width_factor * np.max(points_normalized[:, 0])


def make_a_moved_duplicate(points):
    points = copy.deepcopy(points)
    points, R_exp, t_exp = transform_rigid(points, theta=np.pi/2, t=10)
    # print('Expect R: \n', R_exp, 't\n', t_exp)
    # print('initial moving point cloud center is ', np.mean(points, axis=0))
    return points, R_exp, t_exp


def main():
    """
    test make_up_data functionality
    :return:
    """

    """read bunny data and make a moved copy"""
    points = read_data.read_data()
    points_, _, _ = make_a_moved_duplicate(points)

    """add noise for both, even it doesn't make sense in really application"""
    points = add_Gaussian_noise(points)
    points_ = add_Gaussian_noise(points_)

    """show layout with plt and open3d, oped3d has faster user integration with bunny"""
    vis.vis_plt([points, points_])
    vis.vis_open3d([points, points_])

    # points = np.zeros((10000, 3))
    # # points = np.random.random((10000, 3))
    # # noise = np.random.normal(loc=[0.0, 10.0], scale=[1.0, 4.0], size=points.shape)
    # print('Before add noise, points has center of ', np.mean(points, axis=0), 'std of ', np.std(points, axis=0))
    # points = add_Gaussian_noise(points)
    # print('After add noise, points has center of ', np.mean(points, axis=0), 'std of ', np.std(points, axis=0))


if __name__ == '__main__':
    main()
