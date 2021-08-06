#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   transform.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/3/21 2:32 PM   StrikerCC    1.0         None
'''

# import lib
import numpy as np

# from data import read_data
from utils import rms_error

# global variables
dimension = 3


def find_best_rigid_transform(pc_target, pc_src):
    assert isinstance(pc_target, np.ndarray) and isinstance(pc_src, np.ndarray)
    assert pc_target.shape == pc_src.shape, 'point target shape doesn\'t match ' + str(pc_target.shape) + ' source shape' + str(pc_src.shape)
    assert pc_target.shape[1] == pc_src.shape[1] == dimension
    # compute H
    H = np.matmul(pc_src.T, pc_target)
    U, S, V_T = np.linalg.svd(H)
    R = np.matmul(V_T.T, U.T)
    t = np.mean(pc_target, axis=0).T - np.matmul(R, np.mean(pc_src, axis=0).T)
    t = np.expand_dims(t, axis=-1)
    # if np.isclose(np.linalg.det(R), 1.0, atol=0.001):
    #     I = np.eye(3)
    #     R = 0
    # assert np.isclose(np.linalg.det(R), 1.0, atol=0.001), 'expect det of rotation matrix to be 1, but get ' + str(np.linalg.det(R)) + ' instead'
    assert R.shape == (3, 3) and t.shape == (3, 1),  str(R.shape) + ' ' + str(t.shape)
    return R, t


def rigid(theta=np.pi / 3):
    V_a = np.eye(3)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    V_b = np.matmul(rot, V_a)
    print(V_b, '\n', rot, '\n')

    print(np.matmul(V_a, V_b), '\n')


def transform_rigid(p1, theta=np.pi / 3, t=1):
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    return np.transpose(np.matmul(rot, p1.T) + t), rot.T, -t


def main():
    """"""
    # rotation()

    """read data"""
    array_fix, _, _ = read_data(show=False)
    errors = []
    for i in range(10):
        """make pseudo data"""
        array_move, R_exp, t_exp = transform_rigid(array_fix, theta=np.pi, t=i)

        # print('Expect R: \n', R_exp, 't\n', t_exp)

        """vis"""
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(array_fix[:, 0], array_fix[:, 1], array_fix[:, 2])
        ax.scatter(array_move[:, 0], array_move[:, 1], array_move[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

        """compute best transform"""
        R, t = find_best_rigid_transform(pc_target=array_fix, pc_src=array_move)
        # print('R det', np.linalg.det(R))
        # print('Computed R:\n', R, 't:\n', t)

        array_move = np.transpose(np.matmul(R, array_move.T) + t)
        errors.append(rms_error(array_fix, array_move))
        print(i, 'distance: ')
        print('rms error is ', errors)

        """vis"""
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


    return


if __name__ == '__main__':
    main()