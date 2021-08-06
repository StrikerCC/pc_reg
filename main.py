# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from data import read_data
from match import find_closet_point
from transform import find_best_rigid_transform, transform_rigid
from utils import rms_error


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def registration():
    """

    :return:
    """

    """read data"""
    array_fix, _, _ = read_data(show=True)
    """make pseudo data"""
    array_move, R_exp, t_exp = transform_rigid(array_fix, theta=np.pi, t=10)
    print('Expect R: \n', R_exp, 't\n', t_exp)
    print('initial moving point cloud center is ', np.mean(array_move, axis=0))

    pc_1, pc_2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

    errors = []
    centers = []
    for i in range(10):
        """find point correspondence"""
        index_fix = find_closet_point(array_fix, array_move, k=2)
        index_fix = index_fix[:, 0]
        """compute best transform"""
        R, t = find_best_rigid_transform(pc_target=array_fix[index_fix], pc_src=array_move)
        array_move = np.transpose(np.dot(R, array_move.T) + t)

        """vis"""
        print('R det', np.linalg.det(R))
        print('Computed R:\n', R, 't:\n', t)
        errors.append(rms_error(array_fix[index_fix], array_move))
        centers.append(np.mean(array_move, axis=0))
        print(i, 'distance: ')
        print('rms error is ', errors[-1])
        print('moving point cloud center is ', centers[-1])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(array_fix[:, 0], array_fix[:, 1], array_fix[:, 2])
        ax.scatter(array_move[:, 0], array_move[:, 1], array_move[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    """vis error"""
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(errors)
    plt.show()
        # pc_1.points, pc_2.points = o3d.utility.Vector3dVector(array_fix), o3d.utility.Vector3dVector(array_move)
        # o3d.visualization.draw_geometries([pc_2, pc_1])


def main():
    registration()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
