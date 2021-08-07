#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   reg_open3d.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/6/21 1:52 PM   StrikerCC    1.0         None
'''

# import lib
import copy
import os

import numpy as np
import open3d as o3d

import read_data
import process_data
import utils
import transforms3d as t3d

from process_data import prepare_dataset_artificial, prepare_dataset
from vis import draw_registration_result


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, transformation_init=np.identity(4), voxel_size=0.005):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    print(":: Point-to-point ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def test_artificial_data():
    """prepare data"""
    # source, target = utils.prepare_source_and_target_rigid_3d('../../pc_reg/data/bunny.pcd')
    # source, target = utils.prepare_source_and_target_rigid_3d('../../pc_reg/data/bunny.pcd',
    #                                                           orientation=np.deg2rad([0.0, 0.0, 80.0]),
    #                                                           translation=np.array([0.5, 0.0, 0.0]),
    #                                                           n_random=100,
    #                                                           voxel_size=0.01)

    voxel_size = 0.005  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh, orientation_gt, translation_gt = prepare_dataset_artificial(voxel_size=voxel_size)

    """initial align"""
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size=voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)

    """fine align"""
    result_icp = refine_registration(source, target,
                                     source_fpfh, target_fpfh,
                                     result_ransac.transformation,
                                     voxel_size=voxel_size)
    orientation, translation = t3d.euler.mat2euler(result_icp.transformation[:3, :3]), result_icp.transformation[:3, 3]

    """eval and vis"""
    print('Ground truth:\n  orientation(degree):\n', orientation_gt, '\n  translation:\n', translation_gt)
    print('Computed result:\n  orientation(degree):\n', orientation, '\n  translation:\n', translation)
    print('Difference result:\n  orientation(degree):\n', np.rad2deg(np.abs(orientation_gt - orientation)),
          '\n   translation:\n', np.abs(translation - translation_gt))
    draw_registration_result(source_down, target_down, result_icp.transformation)


def test_real_scan_data():
    """prepare data"""
    # source, target = utils.prepare_source_and_target_rigid_3d('../../pc_reg/data/bunny.pcd')
    # source, target = utils.prepare_source_and_target_rigid_3d('../../pc_reg/data/bunny.pcd',
    #                                                           orientation=np.deg2rad([0.0, 0.0, 80.0]),
    #                                                           translation=np.array([0.5, 0.0, 0.0]),
    #                                                           n_random=100,
    #                                                           voxel_size=0.01)
    dataset_path = './data/TUW_TUW_models/TUW_models/'
    for instance in os.listdir(dataset_path):
        model_o3, views_o3, poses = read_data.read_data_tuw(dataset_path=dataset_path, instance=instance)
        for view_o3, pose in zip(views_o3, poses):
            orientation_gt, translation_gt = pose[:3, :3], pose[:3, 3]

            """vis to debug pose"""
            draw_registration_result(source=view_o3, target=model_o3, transformation=pose, window_name='Ground truth')
            draw_registration_result(source=view_o3, target=model_o3, window_name='initial layout')

            voxel_size = 0.005  # means 5cm for this dataset
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source=model_o3,
                                                                                                 target=view_o3,
                                                                                                 voxel_size=voxel_size)

            """initial align"""
            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size=voxel_size)
            print(result_ransac)
            draw_registration_result(source_down, target_down, result_ransac.transformation, window_name='Initial Registration')

            """fine align"""
            result_icp = refine_registration(source, target,
                                             source_fpfh, target_fpfh,
                                             result_ransac.transformation,
                                             voxel_size=voxel_size)
            orientation, translation = t3d.euler.mat2euler(result_icp.transformation[:3, :3]), result_icp.transformation[:3, 3]

            """eval and vis"""
            print('Ground truth:\n  orientation(degree):\n', orientation_gt, '\n  translation:\n', translation_gt)
            print('Computed result:\n  orientation(degree):\n', orientation, '\n  translation:\n', translation)
            print('Difference result:\n  orientation(degree):\n', np.rad2deg(np.abs(orientation_gt - orientation)),
                  '\n   translation:\n', np.abs(translation - translation_gt))
            draw_registration_result(source_down, target_down, result_icp.transformation, window_name='ICP Registration')
            break

def main():
    """
    use open3d package to register two point cloud
    :return:
    """
    # test_artificial_data()
    test_real_scan_data()


    # for i in range(icp_iteration):
    #     reg_p2p = o3.pipelines.registration.registration_icp(result, target, threshold,
    #                 np.identity(4), o3.pipelines.registration.TransformationEstimationPointToPoint(),
    #                 o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
    #     result.transform(reg_p2p.transformation)
    #     vis.update_geometry(source)
    #     vis.update_geometry(target)
    #     vis.update_geometry(result)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     if save_image:
    #         vis.capture_screen_image("image_%04d.jpg" % i)
    # vis.run()


if __name__ == '__main__':
    main()
