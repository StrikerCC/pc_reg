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
import copy
import open3d as o3
import numpy as np
from utils import estimate_normals
import transforms3d as t3d


def prepare_source_and_target_rigid_3d(source_filename,
                                       noise_amp=0.001,
                                       n_random=500,
                                       orientation=np.deg2rad([0.0, 0.0, 90.0]),
                                       translation=np.ones(3)*0.001,
                                       voxel_size=0.005,
                                       normals=False):
    source = o3.io.read_point_cloud(source_filename)
    # source = source.voxel_down_sample(voxel_size=voxel_size)
    print(source)
    target = copy.deepcopy(source)
    tp = np.asarray(target.points)
    np.random.shuffle(tp)
    rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))
    rands = (np.random.rand(n_random, 3) - 0.5) * rg + tp.mean(axis=0)
    target.points = o3.utility.Vector3dVector(np.r_[tp + noise_amp * np.random.randn(*tp.shape), rands])
    ans = np.identity(4)
    ans[:3, :3] = t3d.euler.euler2mat(*orientation)
    ans[:3, 3] = translation
    target.transform(ans)
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


def prepare_dataset_artificial(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("./data/ICP/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("./data/ICP/cloud_bin_1.pcd")
    orientation = np.deg2rad([0.0, 0.0, 80.0])
    translation = np.array([0.5, 0.0, 0.0])
    source, target = prepare_source_and_target_rigid_3d('./data/bunny.pcd',
                                                              orientation=orientation,
                                                              translation=translation,
                                                              n_random=100,
                                                              voxel_size=0.01)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh, orientation, translation

