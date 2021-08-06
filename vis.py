#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vis.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/5/21 1:21 PM   StrikerCC    1.0         None
'''

# import lib
import matplotlib.pyplot as plt
import open3d as o3d
# from data import read_data


def vis_plt(arrays):
    """vis"""
    if not isinstance(arrays, list):
        arrays = [arrays]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for array in arrays:
        ax.scatter(array[:, 0], array[:, 1], array[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    plt.show()


def vis_open3d(arrays):
    """vis"""
    if not isinstance(arrays, list):
        arrays = [arrays]
    pcs = []
    for array in arrays:
        pcs.append(o3d.geometry.PointCloud())
        pcs[-1].points = o3d.utility.Vector3dVector(array)
    o3d.visualization.draw_geometries(pcs)


def main():
    array_fix, array_move, array_move_reorder = read_data(show=False)
    vis_plt([array_fix, array_move, array_move_reorder])
    vis_open3d([array_fix, array_move, array_move_reorder])


if __name__ == '__main__':
    main()
