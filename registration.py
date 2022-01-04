# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/19/21 5:32 PM
"""
import time

import numpy as np
import transforms3d as t3d
import open3d as o3

from dataset import dataset
from vis import draw_registration_result


TRANSLATION_FAILURE_TOLERANCE = 3.0
ORIENTATION_FAILURE_TOLERANCE = 1.0


def ransac_icp(dataloader, statistic, show_flag=False):
    VOXEL_SIZE_GLOBAL = 5
    # VOXEL_SIZE_LOCAL = 3

    # read source and target pc
    for i in range(len(dataloader)):
        source = dataloader[i]
        pose_gt = source['pose']
        orientation_gt, translation_gt = np.asarray(t3d.euler.mat2euler(pose_gt[:3, :3])), pose_gt[:3, 3]

        # pc_src, pc_tar = o3.io.read_point_cloud(source['pc_model']), o3.io.read_point_cloud(source['pc_artificial'])
        pc_src, pc_tar = source['pc_model'], source['pc_artificial']

        # preprocessing include, down sampling, feature computation, tree building
        time_0 = time.time()
        pc_src, pc_tar = pc_src.voxel_down_sample(VOXEL_SIZE_GLOBAL), pc_tar.voxel_down_sample(VOXEL_SIZE_GLOBAL)
        radius_normal = VOXEL_SIZE_GLOBAL * 2
        pc_src.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = VOXEL_SIZE_GLOBAL * 5
        pc_src_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        pc_tar_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_tar, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))

        # global registration
        transformation_init = np.eye(4)
        distance_threshold = VOXEL_SIZE_GLOBAL * 1.5
        result_global = o3.pipelines.registration.registration_ransac_based_on_feature_matching(
            source=pc_src, target=pc_tar, source_feature=pc_src_fpfh, target_feature=pc_tar_fpfh, mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=4,
            checkers=[o3.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      o3.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            criteria=o3.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999)
        )
        time_global = time.time() - time_0

        # local registration
        time_0 = time.time()
        transformation_init = result_global.transformation
        distance_threshold = VOXEL_SIZE_GLOBAL * 0.4
        result_local = o3.pipelines.registration.registration_icp(
            source=pc_src, target=pc_tar, max_correspondence_distance=distance_threshold, init=transformation_init,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPlane(),
            # criteria=
        )
        time_local = time.time() - time_0

        # record statics
        tf_final = result_local.transformation
        orientation, translation = t3d.euler.mat2euler(tf_final[:3, :3]), tf_final[:3, 3]
        error_t, error_o = np.rad2deg(np.abs(orientation_gt - orientation)), np.abs(translation_gt - translation)
        error_t, error_o = np.linalg.norm(error_t), np.linalg.norm(error_o)
        # # if differ from gt to much, count it as failure, not going to error statics
        statistic['#case'] += 1
        statistic['time_global'] += time_global
        statistic['time_local'] += time_local
        if error_o > ORIENTATION_FAILURE_TOLERANCE or error_t > TRANSLATION_FAILURE_TOLERANCE:
            statistic['#failure'] += 1
        else:
            statistic['error_t'] += error_t
            statistic['error_o'] += error_o

        # vis
        if show_flag:
            # print(time_global, time_local)
            draw_registration_result(source=pc_src, target=pc_tar)
            draw_registration_result(source=pc_src, target=pc_tar, transformation=tf_final)

    statistic['time_global'] /= statistic['#case']
    statistic['time_local'] /= statistic['#case']
    statistic['error_t'] /= (statistic['#case'] - statistic['#failure'])
    statistic['error_o'] /= (statistic['#case'] - statistic['#failure'])

    """output"""
    print('ransac_icp')
    print('Translation rms', statistic['error_t'])
    print('Orientation rms', statistic['error_o'])
    print('Time average', (statistic['time_global'] + statistic['time_local']))
    print('Failure percent', statistic['#failure'] / statistic['#case'])
    return statistic


def fgr_icp(dataloader, statistic, show_flag=False):
    VOXEL_SIZE_GLOBAL = 5
    # VOXEL_SIZE_LOCAL = 3

    # read source and target pc
    for i in range(len(dataloader)):
        source = dataloader[i]
        pose_gt = source['pose']
        orientation_gt, translation_gt = np.asarray(t3d.euler.mat2euler(pose_gt[:3, :3])), pose_gt[:3, 3]

        # pc_src, pc_tar = o3.io.read_point_cloud(source['pc_model']), o3.io.read_point_cloud(source['pc_artificial'])
        pc_src, pc_tar = source['pc_model'], source['pc_artificial']

        # preprocessing include, down sampling, feature computation, tree building
        time_0 = time.time()
        pc_src, pc_tar = pc_src.voxel_down_sample(VOXEL_SIZE_GLOBAL), pc_tar.voxel_down_sample(VOXEL_SIZE_GLOBAL)
        radius_normal = VOXEL_SIZE_GLOBAL * 2
        pc_src.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pc_tar.estimate_normals(o3.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = VOXEL_SIZE_GLOBAL * 5
        pc_src_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_src, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))
        pc_tar_fpfh = o3.pipelines.registration.compute_fpfh_feature(pc_tar, o3.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature, max_nn=100))

        # global registration
        transformation_init = np.eye(4)
        distance_threshold = VOXEL_SIZE_GLOBAL * 1.5
        result_global = o3.pipelines.registration.registration_fast_based_on_feature_matching(
            source=pc_src, target=pc_tar, source_feature=pc_src_fpfh, target_feature=pc_tar_fpfh, mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=4,
            checkers=[o3.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      o3.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            criteria=o3.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.999)
        )
        time_global = time.time() - time_0

        # local registration
        time_0 = time.time()
        transformation_init = result_global.transformation
        distance_threshold = VOXEL_SIZE_GLOBAL * 0.4
        result_local = o3.pipelines.registration.registration_icp(
            source=pc_src, target=pc_tar, max_correspondence_distance=distance_threshold, init=transformation_init,
            estimation_method=o3.pipelines.registration.TransformationEstimationPointToPlane(),
            # criteria=
        )
        time_local = time.time() - time_0

        # record statics
        tf_final = result_local.transformation
        orientation, translation = t3d.euler.mat2euler(tf_final[:3, :3]), tf_final[:3, 3]
        error_t, error_o = np.rad2deg(np.abs(orientation_gt - orientation)), np.abs(translation_gt - translation)
        error_t, error_o = np.linalg.norm(error_t), np.linalg.norm(error_o)
        # # if differ from gt to much, count it as failure, not going to error statics
        statistic['#case'] += 1
        statistic['time_global'] += time_global
        statistic['time_local'] += time_local
        if error_o > ORIENTATION_FAILURE_TOLERANCE or error_t > TRANSLATION_FAILURE_TOLERANCE:
            statistic['#failure'] += 1
        else:
            statistic['error_t'] += error_t
            statistic['error_o'] += error_o

        # vis
        if show_flag:
            # print(time_global, time_local)
            draw_registration_result(source=pc_src, target=pc_tar)
            draw_registration_result(source=pc_src, target=pc_tar, transformation=tf_final)

    statistic['time_global'] /= statistic['#case']
    statistic['time_local'] /= statistic['#case']
    statistic['error_t'] /= (statistic['#case'] - statistic['#failure'])
    statistic['error_o'] /= (statistic['#case'] - statistic['#failure'])

    """output"""
    print('rgr_icp')
    print('Translation rms', statistic['error_t'])
    print('Orientation rms', statistic['error_o'])
    print('Time average', (statistic['time_global'] + statistic['time_local']))
    print('Failure percent', statistic['#failure'] / statistic['#case'])
    return statistic


registrations = [ransac_icp, fgr_icp]

