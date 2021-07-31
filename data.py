#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data.py    
@Contact :   chengc0611@gmail.com
@License :   

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
7/29/2021 2:13 PM   Cheng Chen    1.0         None
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk
import itk


def reorder_data(points):
    index = np.argsort(points, axis=0)
    index_x = index[:, 0]
    points = points[index_x, :]
    return points


def read_data_itk(reorder=False, show=False):
    dimension = 3
    sparse = 10
    file_fix, file_move = r'./data/bunny.txt', r'./data/bunny-Copy.txt'
    array_fix_org, array_move_org = np.loadtxt(file_fix), np.loadtxt(file_move)
    array_fix, array_move = array_fix_org[::sparse], array_move_org[::sparse]
    array_move_reorder = array_move if not reorder else reorder_data(array_move)

    PointSetType = itk.PointSet[itk.F, dimension]

    """itk point type"""
    fixed_points = PointSetType.New()
    moving_points_reorder = PointSetType.New()
    fixed_points_match = PointSetType.New()

    fixed_points.Initialize()
    moving_points_reorder.Initialize()
    fixed_points_match.Initialize()

    for i in range(len(array_fix)):
        fixed_points.SetPoint(i, array_fix[i, :])
        moving_points_reorder.SetPoint(i, array_move_reorder[i, :])
        fixed_points_match.SetPoint(i, array_move[i, :])
    """vis"""
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(array_fix[:, 0], array_fix[:, 1], array_fix[:, 2])
        # ax.scatter(array_move[:, 0], array_move[:, 1], array_move[:, 2])
        ax.scatter(array_move_reorder[:, 0], array_move_reorder[:, 1], array_move_reorder[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    print('read ', len(array_fix_org), 'points from txt, using ',  len(array_fix), 'points for registration')
    print('x range ', np.min(array_fix[:, 0]), ' ', np.max(array_fix[:, 0]))
    print('y range ', np.min(array_fix[:, 1]), ' ', np.max(array_fix[:, 1]))
    print('z range ', np.min(array_fix[:, 2]), ' ', np.max(array_fix[:, 2]))
    print('two point sets are', np.mean(np.linalg.norm(array_fix-array_move, axis=1), axis=0), 'away from each other')
    return fixed_points, moving_points_reorder, fixed_points_match


def main():
    read_data_itk(show=True)


if __name__ == '__main__':
    main()