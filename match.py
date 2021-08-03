#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   match.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/2/21 8:10 PM   StrikerCC    1.0         None
'''

# import lib
import time

import numpy as np
from scipy import spatial

import open3d as o3d


def find_correspondences(p1, p2, debug=False):
    assert isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray), 'expect numpy array, but get ' + str(type(p1)) + 'and ' + str(type(p1)) + ' instead '
    assert p1.shape[1] == p2.shape[1] == 3, str(p1.shape) + ' ' + str(p2.shape)

    time_0 = time.time()

    """build tree upon p1, assume p1 is fixed"""
    tree1 = spatial.KDTree(p1)

    """build query"""
    pts = p2

    """In p1, find nearest neighbor of each p2"""
    dis, index = tree1.query(pts, k=5)

    if debug:
        print('scipy Kd-tree take ', time.time() - time_0, 'to find nearest neighbor of ', pts)
        for i in range(p2.shape[1]):
            print('for ', p2[i], ' min dis is ', dis[i], 'index is', index[i], ' neighbor is', p1[index[i]])

    """assume correspondences is nearest neighbor"""
    assert len(index) == len(p2), 'expect ' + str(len(p2)) + ' match points, but get ' + str(index.shape) + ' instead'
    return index


def main():
    x, y = np.mgrid[-10:10, -10:18]
    p1 = np.vstack([x.ravel(), y.ravel(), np.ones(x.size)]).T
    x, y = np.mgrid[0:5, 2:8]
    p2 = np.vstack([x.ravel(), y.ravel(), np.zeros(x.size)]).T

    """visualization"""
    pc_1, pc_2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pc_1.points, pc_2.points = o3d.utility.Vector3dVector(p1), o3d.utility.Vector3dVector(p2)
    o3d.visualization.draw_geometries([pc_2, pc_1])

    p1 = p1[:, np.argsort(p1[1, :])]
    find_correspondences(p1, p2)


if __name__ == '__main__':
    main()
