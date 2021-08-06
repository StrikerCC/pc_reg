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
import numpy as np
import open3d as o3d
import utils
import transforms3d as t3d

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4559,
    #                                   front=[0.6452, -0.3036, -0.7011],
    #                                   lookat=[1.9892, 2.0208, 1.8945],
    #                                   up=[-0.2779, -0.9482, 0.1556])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("./data/ICP/cloud_bin_0.pcd")
    # target = o3d.io.read_point_cloud("./data/ICP/cloud_bin_1.pcd")
    orientation = np.deg2rad([0.0, 0.0, 80.0])
    translation = np.array([0.5, 0.0, 0.0])
    source, target = utils.prepare_source_and_target_rigid_3d('./data/bunny.pcd',
                                                              orientation=orientation,
                                                              translation=translation,
                                                              n_random=100,
                                                              voxel_size=0.01)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh, orientation, translation


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
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
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


def main():
    """
    use open3d package to register two point cloud
    :return:
    """

    """prepare data"""
    # source, target = utils.prepare_source_and_target_rigid_3d('../../pc_reg/data/bunny.pcd')
    # source, target = utils.prepare_source_and_target_rigid_3d('../../pc_reg/data/bunny.pcd',
    #                                                           orientation=np.deg2rad([0.0, 0.0, 80.0]),
    #                                                           translation=np.array([0.5, 0.0, 0.0]),
    #                                                           n_random=100,
    #                                                           voxel_size=0.01)

    voxel_size = 0.005  # means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh, orientation_gt, translation_gt = prepare_dataset(voxel_size)


    """add vis object"""
    # vis = o3.visualization.Visualizer()
    # vis.create_window()
    # result = copy.deepcopy(source)
    # source.paint_uniform_color([1, 0, 0])
    # target.paint_uniform_color([0, 1, 0])
    # result.paint_uniform_color([0, 0, 1])
    # vis.add_geometry(source)
    # vis.add_geometry(target)
    # vis.add_geometry(result)
    # threshold = 0.05
    # icp_iteration = 10000
    # save_image = False

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
    print('Difference result:\n  orientation(degree):\n', np.rad2deg(np.abs(orientation_gt-orientation)), '\n   translation:\n', np.abs(translation-translation_gt))
    draw_registration_result(source_down, target_down, result_icp.transformation)

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
