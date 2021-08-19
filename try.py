#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   try.py    
@Contact :   chengc0611@mgail.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author      @Version    @Desciption
------------      -------      --------    -----------
8/5/21 2:15 PM   StrikerCC    1.0         None
'''

# import lib
import copy

import numpy as np
import open3d as o3d
from vis import draw_registration_result
import transforms3d as t3d


#
# def main():
#     vis1 = o3d.visualization.Visualizer()
#     while True:
#         vis1.create_window(window_name='Input', width=960, height=540, left=0, top=0)
#         vis1.add_geometry(src_pcd_before)
#         vis1.update_geometry(src_pcd_before)


def rigid(src, pose):
    tf = np.eye(4)
    tf[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([30.0, 30.0, 0.0]))
    tf[:3, 3] = 0.01
    pose = np.matmul(tf, pose)
    src.transform(tf)
    return pose

def transform_test():
    src = o3d.io.read_point_cloud('./data/TUW_TUW_models/TUW_models/bunny/3D_model.pcd')
    tar1 = copy.deepcopy(src)
    tar2 = copy.deepcopy(src)

    pose = np.eye(4)
    for _ in range(5):
        pose = rigid(tar1, pose)
    print('current pose', pose)

    tar2.transform(pose)
    o3d.visualization.draw_geometries([tar1, tar2])
    print(src)


def set_test():
    src = ['dog', 'fish', 'human']
    src_set = set(src)
    src.append('bird')
    for s in src:
        print(s, 'in?', s in src_set)


def main():
    set_test()




    # a = np.random.random((10000, 2))
    # noise = np.random.normal(loc=[0.0, 10.0], scale=[1.0, 4.0], size=a.shape)
    # print(a.shape)
    # print(noise.shape)
    # print(np.mean(noise, axis=0))
    # print(np.std(noise, axis=0))


if __name__ == '__main__':
    main()
