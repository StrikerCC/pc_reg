#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   probreg_test.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/6/21 11:17 AM   StrikerCC    1.0         None
'''

# import lib
import copy
import time

import numpy as np
import open3d as o3
# from o3.pipelines.registration import registration_icp as icp
from probreg import cpd, bcpd, l2dist_regs, gmmtree, filterreg
import utils


# load source and target point cloud
source = o3.io.read_point_cloud('./data/bunny.pcd')
target = copy.deepcopy(source)
# transform target point cloud
th = np.deg2rad(30.0)
target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]]))

source, target = utils.prepare_source_and_target_rigid_3d('./data/bunny.pcd')

for reg in [o3.pipelines.registration.registration_icp, cpd.registration_cpd, bcpd.registration_bcpd, gmmtree.registration_gmmtree, filterreg.registration_filterreg]:
# for reg in []:
    time_0 = time.time()
    source = source.voxel_down_sample(voxel_size=0.005)
    target = target.voxel_down_sample(voxel_size=0.005)
    # compute cpd registration
    # tf_param, _, _ = cpd.registration_cpd(source, target)  # , use_cuda=True)

    """registration"""
    if reg == o3.pipelines.registration.registration_icp:
        for _ in range(100):
            res = reg(source, target, max_correspondence_distance=0.04)
    else:
        # for _ in range(10):
        res = reg(source, target)  # , use_cuda=True)

    """format result"""
    if isinstance(res, tuple):
        tf_param = res[0]
    elif reg == o3.pipelines.registration.registration_icp:
        tf_param = res
    else:
        tf_param = res.rigid_trans

    print('time ', time.time() - time_0, ' on ', reg)

    result = copy.deepcopy(source)
    if reg == o3.pipelines.registration.registration_icp:
        result.transform(tf_param.transformation)
        print(tf_param.transformation)
    else:
        result.points = tf_param.transform(result.points)
        print(tf_param.rot, tf_param.t)

    # draw result
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    result.paint_uniform_color([0, 0, 1])
    o3.visualization.draw_geometries([source, target, result])
    o3.visualization.draw_geometries([result])
# o3.visualization.draw_geometries([target, result])
