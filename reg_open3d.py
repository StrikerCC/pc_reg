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
import time

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import read_data
import process_data
import utils
import transforms3d as t3d

from process_data import prepare_dataset_artificial, prepare_dataset
from vis import draw_registration_result


def ransac_global_registration(source_down, target_down, source_fpfh,
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


def fast_global_registration(source_down, target_down, source_fpfh,
                             target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


# def registration_filterreg(source_down, target_down, source_fpfh,
#                            target_fpfh, voxel_size,target_normals=None,
#                            sigma2=None, update_sigma2=False, w=0, objective_type='pt2pt', maxiter=50,
#                            tol=0.001, min_sigma2=1.0e-4, feature_fn=lambda x: x,
#                            callbacks=[], **kargs):
#     """FilterReg registration
#
#     Args:
#         source_down (numpy.ndarray): Source point cloud data.
#         target_down (numpy.ndarray): Target point cloud data.
#         target_normals (numpy.ndarray, optional): Normal vectors of target point cloud.
#         sigma2 (float, optional): Variance of GMM. If `sigma2` is `None`, `sigma2` is automatically updated.
#         w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
#         objective_type (str, optional): The type of objective function selected by 'pt2pt' or 'pt2pl'.
#         maxitr (int, optional): Maximum number of iterations to EM algorithm.
#         tol (float, optional): Tolerance for termination.
#         min_sigma2 (float, optional): Minimum variance of GMM.
#         feature_fn (function, optional): Feature function. If you use FPFH feature, set `feature_fn=probreg.feature.FPFH()`.
#         callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
#             `callback(probreg.Transformation)`
#
#     Keyword Args:
#         tf_init_params (dict, optional): Parameters to initialize transformation (for rigid).
#     """
#     cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
#     frg = RigidFilterReg(cv(source_down), cv(target_normals), sigma2, update_sigma2, **kargs)
#     frg.set_callbacks(callbacks)
#     return frg.registration(cv(target_down), w=w, objective_type=objective_type, maxiter=maxiter,
#                             tol=tol, min_sigma2=min_sigma2, feature_fn=feature_fn)


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


def test_artificial_data(flag_show=True, flag_cout=True):
    """"""
    """name global registration"""
    global_registrations = [ransac_global_registration, fast_global_registration]
    statistics = { global_registration: {'method': global_registration,
                                          'model': [],
                                          's#': [],
                                          't#': [],
                                          'r': [],
                                          't': [],
                                          'time_global': [],
                                          'time_local': []} for global_registration in global_registrations}

    """prepare data"""
    dataset_path = './data/TUW_TUW_models/TUW_models/'
    instances = os.listdir(dataset_path)
    source_filenames = [dataset_path + instance + '/3D_model.pcd' for instance in instances]
    source_filenames.append('../OverlapPredator/data/stanford/bunny.pcd')
    source_filenames.reverse()
    # source, target = utils.prepare_source_and_target_rigid_3d('../../pc_reg/data/bunny.pcd')
    # source, target = utils.prepare_source_and_target_rigid_3d('../../pc_reg/data/bunny.pcd',
    #                                                           orientation=np.deg2rad([0.0, 0.0, 80.0]),
    #                                                           translation=np.array([0.5, 0.0, 0.0]),
    #                                                           n_random=100,
    #                                                           voxel_size=0.01)

    voxel_size = 0.005  # means 5cm for this dataset

    for source_filename in source_filenames:
        for global_registration in global_registrations:
            time_0 = time.time()
            source, target, source_down, target_down, source_fpfh, target_fpfh, orientation_gt, translation_gt = \
                prepare_dataset_artificial(source_filename=source_filename, voxel_size=voxel_size)
            if flag_show:
                draw_registration_result(source=source, target=target, window_name='initial layout')

            """initial align"""
            result_global = global_registration(source_down=source_down, target_down=target_down,
                                                source_fpfh=source_fpfh, target_fpfh=target_fpfh,
                                                voxel_size=voxel_size)
            time_global = time.time() - time_0
            # print(str(global_registration), 'take', )  # result_ransac)
            if flag_show:
                draw_registration_result(source=source_down, target=target_down,
                                         transformation=result_global.transformation,
                                         window_name='Initial Registration')

            """fine align"""
            time_0 = time.time()
            result_icp = refine_registration(source=source_down, target=target_down,
                                             source_fpfh=source_fpfh, target_fpfh=target_fpfh,
                                             transformation_init=result_global.transformation,
                                             voxel_size=voxel_size)
            time_local = time.time() - time_0
            orientation, translation = t3d.euler.mat2euler(result_icp.transformation[:3, :3]), result_icp.transformation[:3, 3]
            # t error
            rms_error_t = translation_gt - translation
            rms_error_t = np.linalg.norm(rms_error_t) * 1000
            # rotation error
            rms_error_r = orientation_gt - orientation
            rms_error_r = np.linalg.norm(rms_error_r)

            # result statistics
            assert statistics[global_registration]['method'] == global_registration, str(statistics[global_registration]['method']) + ' is not ' + str(global_registration)
            statistics[global_registration]['model'].append(source_filename)
            statistics[global_registration]['s#'].append(len(source.points))
            statistics[global_registration]['t#'].append(len(target.points))
            statistics[global_registration]['r'].append(np.rad2deg(rms_error_r))
            statistics[global_registration]['t'].append(rms_error_t)
            statistics[global_registration]['time_global'].append(time_global)
            statistics[global_registration]['time_local'].append(time_local)

            """eval and vis"""
            if flag_cout:
                print(global_registration, source_filename)
                print('Finished in', time_global + time_local)
                print('Ground truth:\n  orientation(degree):\n', np.rad2deg(orientation_gt), '\n  translation:\n',
                      translation_gt)
                print('Computed result:\n  orientation(degree):\n', np.rad2deg(orientation), '\n  translation:\n',
                      translation)
                print('Difference result:\n  orientation(degree):\n', statistics[global_registration]['r'][-1],
                      '\n   translation:\n', statistics[global_registration]['t'][-1])
            if flag_show:
                draw_registration_result(source_down, target_down, result_icp.transformation,
                                         window_name='Final Registration')

        # statistics.sort(key=lambda error: np.linalg.norm(error['t']))

    if flag_cout:
        for reg in statistics.keys():
            for i, id in enumerate(statistics[reg]['model']):
                print('     ', reg, id, '\n         ', statistics[reg]['time_global'][i] + statistics[reg]['time_global'][i])

    """plot"""
    fig = plt.figure()
    # plot translation error
    ax_translation = fig.add_subplot(3, 1, 1)
    for key in statistics.keys():
        ax_translation.plot(statistics[key]['t'], label=key)
    ax_translation.set_xlabel('model id')
    ax_translation.set_ylabel('RMS translation error (mm)')
    ax_translation.legend()

    # plot rotation error
    ax_r = fig.add_subplot(3, 1, 2)
    for key in statistics.keys():
        ax_r.plot(statistics[key]['r'], label=key)
    ax_r.set_xlabel('model id')
    ax_r.set_ylabel('RMS rotation error (degree)')
    ax_translation.legend()

    # plot number of points
    ax_time = fig.add_subplot(3, 1, 3)
    for key in statistics.keys():
        ax_time.plot(statistics[key]['time_global'] + statistics[key]['time_local'], label=key)
    ax_time.set_xlabel('model id')
    ax_time.set_ylabel('Time cost (seconds)')
    ax_translation.legend()

    plt.show()


def test_real_scan_data():
    """"""
    """name global registration"""
    global_registrations = [ransac_global_registration, fast_global_registration]
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
            source, target = copy.deepcopy(view_o3), copy.deepcopy(model_o3)
            orientation_gt, translation_gt = pose[:3, :3], pose[:3, 3]

            """vis to debug pose"""
            draw_registration_result(source=source, target=target, transformation=pose, window_name='Ground truth')
            draw_registration_result(source=source, target=target, window_name='initial layout')

            voxel_size = 0.005  # means 5cm for this dataset
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source=source,
                                                                                                 target=target,
                                                                                                 voxel_size=voxel_size)

            for global_registration in global_registrations:
                """initial align"""
                result_ransac = global_registration(source_down=source_down, target_down=target_down,
                                                    source_fpfh=source_fpfh, target_fpfh=target_fpfh,
                                                    voxel_size=voxel_size)
                print(str(global_registration), result_ransac)
                transform_global = result_ransac.transformation
                # draw_registration_result(source_down, target_down, transform_global, window_name='Initial Registration')
                draw_registration_result(source=source_down, target=target_down, transformation=transform_global,
                                         window_name='Initial Registration')

                """fine align"""
                result_icp = refine_registration(source, target,
                                                 source_fpfh, target_fpfh,
                                                 result_ransac.transformation,
                                                 voxel_size=voxel_size)
                transform_local = result_icp.transformation
                orientation, translation = t3d.euler.mat2euler(transform_local[:3, :3]), transform_local[:3, 3]

                """eval and vis"""
                print('Ground truth:\n  orientation(degree):\n', orientation_gt, '\n  translation:\n', translation_gt)
                print('Computed result:\n  orientation(degree):\n', orientation, '\n  translation:\n', translation)
                print('Difference result:\n  orientation(degree):\n', np.rad2deg(np.abs(orientation_gt - orientation)),
                      '\n   translation:\n', np.abs(translation - translation_gt))
                # draw_registration_result(source_down, target_down, transform_local, window_name='ICP Registration')
                draw_registration_result(source=source_down, target=target_down, transformation=transform_local,
                                         window_name='Initial Registration')
            break


def main():
    """
    use open3d package to register two point cloud
    :return:
    """
    test_artificial_data(flag_show=True, flag_cout=True)
    # test_real_scan_data()

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
