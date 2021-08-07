#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/3/21 2:33 PM   StrikerCC    1.0         None
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt
import copy
import open3d as o3
import transforms3d as t3d


def ordinary_least_square(A, b):
    """
    solve this by ordinary_least_square, give a combination of columns in A to be projection of b on A (
    :param A: space
    :param b: desired projection
    :return:
    """
    assert isinstance(A, np.ndarray) and isinstance(b, np.ndarray)
    assert A.shape[0] > A.shape[1] and len(A.shape) == 2, A.shape
    assert A.shape[0] == b.shape[0], str(A.shape) + ', ' + str(b.shape)
    #                <<<<<<<<<<<<<<<< pseudo-inverse >>>>>>>>>>>>>>>
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)   # compute the pseudo-inverse


def Ridge_regression(A, b, lambda_):
    """
    solve the a the combination of columns in A to be projection of b on A
    :param A: space
    :param b: desired projection
    :return:
    """
    assert isinstance(A, np.ndarray) and isinstance(b, np.ndarray)
    assert A.shape[0] > A.shape[1] and len(A.shape) == 2,  A.shape
    assert A.shape[0] == b.shape[0], str(A.shape) + ', ' + str(b.shape)
    return np.matmul(np.linalg.inv(np.matmul(A.T, A)) + lambda_ * np.eye(A.shape[1]), np.matmul(A.T, b))   # compute the pseudo-inverse


def solve_null_space(A):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2, A.shape
    assert np.linalg.matrix_rank(A) == min(A.shape) - 1, 'null space is more than one dimension'

    return


def rms_error(data_a, data_b):
    assert isinstance(data_a, np.ndarray) and isinstance(data_b, np.ndarray)
    assert data_a.shape == data_b.shape
    assert len(data_a.shape) == len(data_b.shape) == 2

    # transpose the data if number of rows is bigger than number of cols
    if data_a.shape[0] < data_a.shape[1]:
        data_a, data_b = data_a.T, data_b.T
    # diff = data_a - data_b
    # diff = np.linalg.norm(diff, axis=1)
    # diff = np.mean(diff, axis=0)
    return np.mean(np.linalg.norm(data_a - data_b, axis=1), axis=0)


def main():
    # make data to fit
    paras = [3.4, -1.5, -2.2, 3.1, 2.1, 1.1]                                       # parameters
    # a, b = 0.1, 0.2                                       # parameters

    t = np.linspace(-100, 100, 100)                                          # time

    for i in range(len(paras)):
        A = np.ones(t.shape) if i == 0 else np.vstack((A, np.power(t, i)))   # time square, time, constant
    A = A.T

    amp = 1
    noise = np.random.random(A.shape)
    A_noise = A + noise * amp

    # para_expect = np.array([a, b, c]).reshape(A.shape[1], 1)
    para_expect = np.array(paras).reshape(A.shape[1], 1)
    proj = np.matmul(A_noise, para_expect)                                              # expect observations

    plt.scatter(t, proj)
    # proj = np.sort(proj, axis=0)
    # plt.scatter(t, proj)
    plt.show()

    # solve linear_least_square
    x_ = ordinary_least_square(A, proj)
    proj_ = np.matmul(A, x_)                                            # observation according to line fitting
    # y_ = np.squeeze(y_, axis=-1).reshape(t.shape[0], -1)
    print(x_)
    plt.scatter(t, proj)
    # plt.show()
    plt.plot(t, proj_)
    plt.show()


def estimate_normals(pcd, params):
    pcd.estimate_normals(search_param=params)
    pcd.orient_normals_to_align_with_direction()


def prepare_source_and_target_nonrigid_2d(source_filename,
                                          target_filename):
    source = np.loadtxt(source_filename)
    target = np.loadtxt(target_filename)
    return source, target


def prepare_source_and_target_nonrigid_3d(source_filename,
                                          target_filename,
                                          voxel_size=5.0):
    source = o3.geometry.PointCloud()
    target = o3.geometry.PointCloud()
    source.points = o3.utility.Vector3dVector(np.loadtxt(source_filename))
    target.points = o3.utility.Vector3dVector(np.loadtxt(target_filename))
    source = source.voxel_down_sample(voxel_size=voxel_size)
    target = target.voxel_down_sample(voxel_size=voxel_size)
    print(source)
    print(target)
    return source, target


if __name__ == "__main__":
    main()

