#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   process_data.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/6/21 5:41 PM   StrikerCC    1.0         None
'''

# import lib
import os
import copy
import open3d as o3
import numpy as np
from utils import estimate_normals
import transforms3d as t3d


def prepare_source_and_target_rigid_3d(source_filename,
                                       noise_amp=0.001,
                                       n_random=500,
                                       orientation=np.deg2rad([0.0, 0.0, 90.0]),
                                       translation=np.ones(3) * 0.01,
                                       voxel_size=0.005,
                                       normals=False):
    """

    :param source_filename:
    :param noise_amp:
    :param n_random:
    :param orientation:
    :param translation:
    :param voxel_size:
    :param normals:
    :return:
    """
    source = o3.io.read_point_cloud(source_filename)
    # source = source.voxel_down_sample(voxel_size=voxel_size)
    print(source)
    if 'bunny' in source_filename:
        ans = np.identity(4)
        ans[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([90.0, 0.0, 0.0]))
        ans[:3, 3] = 0
        source.transform(ans)
    target = copy.deepcopy(source)

    tp = np.asarray(target.points)
    dis_neareast_neighbor = min(np.linalg.norm(tp - tp[0, :], axis=1)[2:])

    """shuffle the data"""
    np.random.shuffle(tp)

    """setup noise"""
    rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))    # range
    rands = (np.random.rand(n_random, 3) - 0.5) * rg + tp.mean(axis=0)

    """add a plane underneath the model"""
    plane_amp = 1.0
    nx = int(plane_amp*rg[0] / dis_neareast_neighbor)
    ny = int(plane_amp*rg[1] / dis_neareast_neighbor)
    x = np.linspace(-plane_amp*rg[0], rg[0]*plane_amp, nx)
    y = np.linspace(-plane_amp*rg[1], rg[1]*plane_amp, ny)
    x, y = np.meshgrid(x, y)

    # make a empty shadow
    mask = np.logical_or(y < - rg[0] / 8, np.logical_or(x < - rg[0] / 4, x > rg[0] / 4))
    x, y = x[mask], y[mask]
    # x = x[]

    z = np.zeros(y.shape) + tp.min(axis=0)[2]
    plane = np.stack([x, y, z])
    plane = np.reshape(plane, newshape=(3, -1)).T

    # make a hole at the intersection and behind
    model_center = np.mean(tp, axis=0)
    dis = np.linalg.norm(plane - model_center, axis=1)
    mask = dis > rg[1] / 2 * 0.75

    tp = np.vstack((tp, plane[mask]))

    target.points = o3.utility.Vector3dVector(np.r_[tp + noise_amp * np.random.randn(*tp.shape), rands])

    ans = np.identity(4)
    ans[:3, :3] = t3d.euler.euler2mat(*orientation)
    ans[:3, 3] = translation
    target.transform(ans)
    print(target)

    if normals:
        estimate_normals(source, o3.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
        estimate_normals(target, o3.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
    return source, target


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def prepare_dataset_artificial(source_filename, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    # source = o3d.io.read_point_cloud("./data/ICP/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("./data/ICP/cloud_bin_1.pcd")
    orientation = np.deg2rad([60.0, 60.0, 60.0])
    translation = np.array([-0.13, -0.13, -0.13])

    source, target = prepare_source_and_target_rigid_3d(source_filename,
                                                        # source, target = prepare_source_and_target_rigid_3d('./data/cloud_0.pcd',
                                                        noise_amp=0.001,
                                                        n_random=100,
                                                        orientation=orientation,
                                                        translation=translation,
                                                        voxel_size=0.01)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh, orientation, translation


def main():
    txt_save_path = 'data/snapshot'
    source, target, *_ = prepare_dataset_artificial(voxel_size=0.005)

    if not os.path.isdir(txt_save_path):
        os.makedirs(txt_save_path)
    # np.savetxt(txt_save_path+'/bunny.txt', source.points)
    # np.savetxt(txt_save_path+'/bunnyCopy.txt', target.points)
    o3.io.write_point_cloud(txt_save_path+'/bunnyBG.pcd', target)


if __name__ == '__main__':
    main()
