#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   reg_itk_v4.py
@Contact :   chengc0611@gmail.com
@License :   

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
7/27/2021 3:42 PM   Cheng CHen    1.0         None
'''

# import lib
import time

import numpy as np
import matplotlib.pyplot as plt

from data import read_data_itk
from itk_test.icp_ED_LM import reg_object_init, reg_object_type_def

dimension = 0


def print_iteration(optimizer):
    print(f'It: {optimizer.GetCurrentIteration()}'
          f' metric value: {optimizer.GetCurrentMetricValue():.6f} ')


def print_point(vals: list) -> str:
    return f'[{",".join(f"{x:.4f}" for x in vals)}]'


def reg(show=False, evulate=False):
    global dimension
    dimension = 3
    # Make point sets
    fixed_set, moving_set, fixed_set_match = read_data_itk(reorder=True, show=True)
    num_iterations = 450
    tolerance = 0.05
    passed = True

    print('initializing metric and optimizer')
    time_0 = time.time()
    TransformType, PointSetMetricType, OptimizerType, RegistrationType = reg_object_type_def()
    transform, optimizer = reg_object_init(TransformType,
                                              PointSetMetricType, fixed_set, moving_set,
                                              OptimizerType, num_iterations,
                                              RegistrationType)
    print('initialized metric and optimizer')

    # optimizer.AddObserver(itk_test.IterationEvent(), print_iteration(optimizer))

    # Run optimization to align the point sets

    # registration.Update()
    optimizer.optimizer()

    print(num_iterations, ' iterations took ', time.time() - time_0, 'seconds, which is ',
          (time.time() - time_0) / 60.0, 'minutes')
    print(f'Number of iterations: {num_iterations}')
    # print(f'Moving-source final value: {optimizer.GetCurrentMetricValue()}')
    # print(f'Moving-source final position: {list(optimizer.GetCurrentPosition())}')
    # print(f'Optimizer scales: {list(optimizer.GetScales())}')
    # print(f'Optimizer learning rate: {optimizer.GetLearningRate()}')

    # applying the resultant transform to moving points and verify result
    print('Fixed\tMoving\tMovingTransformed\tFixedTransformed\tDiff')
    transform.GetParameters()

    print(num_iterations, ' iterations took ', time.time() - time_0, 'seconds, which is ',
          (time.time() - time_0) / 60.0, 'minutes')

    if evulate:
        list_fixed, list_moved, differences = [], [], []

        # for n in range(0, metric.GetNumberOfComponents()):
        #     transformed_moving_point = moving_inverse.TransformPoint(moving_set.GetPoint(n))
        #     transformed_fixed_point = fixed_inverse.TransformPoint(fixed_set.GetPoint(n))
        #
        #     # assume two point set match
        #     difference = [transformed_moving_point[dim] - transformed_fixed_point[dim]
        #                   for dim in range(0, dimension)]
        #
        #     list_fixed.append([coord for coord in transformed_fixed_point])
        #     list_moved.append([coord for coord in transformed_moving_point])
        #     differences.append(difference)
        #     # print(f'{print_point(fixed_set.GetPoint(n))}'
        #     #       f'\t{print_point(moving_set.GetPoint(n))}'
        #     #       f'\t{print_point(transformed_moving_point)}'
        #     #       f'\t{print_point(transformed_fixed_point)}'
        #     #       f'\t{print_point(difference)}')
        #
        #     if (any(abs(difference[dim]) > tolerance
        #             for dim in range(0, dimension))):
        #         passed = False

        """vis"""
        if show:
            array_fixed, array_moved = np.array(list_fixed), np.array(list_moved)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(array_fixed[:, 0], array_fixed[:, 1], array_fixed[:, 2])
            ax.scatter(array_moved[:, 0], array_moved[:, 1], array_moved[:, 2])
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()

        """statistic"""
        differences = np.array(differences)
        difference_mean = np.mean(differences, axis=0)
        difference_std = np.std(differences, axis=0)
        print('root mean square error, ', np.mean(np.linalg.norm(differences, axis=1), axis=0))

        if not passed:
            raise Exception('Transform outside of allowable tolerance')
        else:
            print('Transform is within allowable tolerance.')


def main():
    reg(show=True, evulate=True)


if __name__ == '__main__':
    main()
