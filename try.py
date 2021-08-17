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
import numpy as np
import open3d as o3d
from vis import draw_registration_result

#
# def main():
#     vis1 = o3d.visualization.Visualizer()
#     while True:
#         vis1.create_window(window_name='Input', width=960, height=540, left=0, top=0)
#         vis1.add_geometry(src_pcd_before)
#         vis1.update_geometry(src_pcd_before)


def main():
    for i, file in enumerate(['./data/TUW_TUW_models/TUW_models/bunny/',
                              'data/TUW_TUW_models/TUW_models/dragon_recon/',
                              'data/TUW_TUW_models/TUW_models/happy_recon/',
                              'data/TUW_TUW_models/TUW_models/lucy/',
                              'data/TUW_TUW_models/TUW_models/dragon_xyz/',
                              ]):
        pc = None
        if i == 0:
            pc = o3d.io.read_point_cloud(file + 'reconstruction/bun_zipper.ply')
        if i == 1:
            pc = o3d.io.read_point_cloud(file + 'dragon_vrip.ply')
        if i == 2:
            pc = o3d.io.read_point_cloud(file + 'happy_vrip.ply')
        if i == 3:
            pc = o3d.io.read_point_cloud(file + 'lucy.ply')
        # if i == 4:
        #     pc = o3d.io.read_point_cloud(file + 'data.txt')

        draw_registration_result(pc)
        o3d.io.write_point_cloud(file + '3D_model.pcd', pc)

    # a = np.random.random((10000, 2))
    # noise = np.random.normal(loc=[0.0, 10.0], scale=[1.0, 4.0], size=a.shape)
    # print(a.shape)
    # print(noise.shape)
    # print(np.mean(noise, axis=0))
    # print(np.std(noise, axis=0))


if __name__ == '__main__':
    main()
