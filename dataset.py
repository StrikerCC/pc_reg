# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/13/21 5:44 PM
"""

import copy
import os
import glob
import numpy as np
import json

import open3d as o3
import transforms3d as t3d
from vis import draw_registration_result


class dataset:
    def __init__(self):
        self.flag_show = True
        self.voxel_sizes = (0.10, 0.273 * 2)
        self.voxel_sizes = np.arange(self.voxel_sizes[0], self.voxel_sizes[1], (self.voxel_sizes[1]-self.voxel_sizes[0]) / 100)

        self.angle_cutoff_along = (0, 360)
        self.angle_cutoff_along = np.arange(self.angle_cutoff_along, self.angle_cutoff_along, 45)
        self.angle_cutoff = 90

        self.plane_sizes = (1.5, 2.5)
        self.plane_sizes = np.arange(self.plane_sizes[0], self.plane_sizes[1], (self.plane_sizes[1]-self.plane_sizes[0]) / 100)

        self.Gaussian_sigma_factor = (0.2, 1.2)
        self.Gaussian_sigma_factor = np.arange(self.Gaussian_sigma_factor[0], self.Gaussian_sigma_factor[1], (self.Gaussian_sigma_factor[1]-self.Gaussian_sigma_factor[0])/100)

        self.num_random = np.arange(50, 250, 100)
        self.dir_path_read = None
        self.file_paths = None

    def read(self, dir_path):
        self.dir_path_read = dir_path
        instances = os.listdir(dir_path)
        self.file_paths = [dir_path + instance + '/3D_model.pcd' for instance in instances]

    def generate(self, dir_path):
        for file_path in self.file_paths:
            source = o3.io.read_point_cloud(filename=file_path)
            if 'bunny' in file_path:
                ans = np.identity(4)
                ans[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([90.0, 0.0, 0.0]))
                ans[:3, 3] = 0
                source.transform(ans)
            target = copy.deepcopy(source)

    def add_noise(self, pcs):
        pcs_processed = []
        for pc in pcs:
            tp = np.asarray(pc.points)
            tp = copy.deepcopy(tp)
            """setup noise"""
            # rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))  # range
            for noise_amp in self.Gaussian_sigma_factor:
                pc_ = copy.deepcopy(pc)
                pc_.points = o3.utility.Vector3dVector(np.r_[tp + noise_amp * np.random.randn(*tp.shape)])
                pcs_processed.append(pc_)
                if self.flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = 1
                    draw_registration_result(source=pc, target=pcs_processed[-1], transformation=tf, window_name='Initial Setup')
        return pcs_processed

    def add_plane(self, pcs):
        pcs_processed = []
        for pc in pcs:
            tp = np.asarray(pc.points)
            tp = copy.deepcopy(tp)
            dis_nearest_neighbor = min(np.linalg.norm(tp - tp[0, :], axis=1)[2:])
            rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))  # range

            for plane_size in self.plane_sizes:
                pc_ = copy.deepcopy(pc)

                # add a plane underneath the model
                plane_amp = plane_size
                nx = int(plane_amp * rg[0] / dis_nearest_neighbor)
                ny = int(plane_amp * rg[1] / dis_nearest_neighbor)
                x = np.linspace(-plane_amp * rg[0], rg[0] * plane_amp, nx)
                y = np.linspace(-plane_amp * rg[1], rg[1] * plane_amp, ny)
                x, y = np.meshgrid(x, y)

                # make a empty shadow
                mask = np.logical_or(y < - rg[0] / 8, np.logical_or(x < - rg[0] / 4, x > rg[0] / 4))
                x, y = x[mask], y[mask]
                z = np.zeros(y.shape) + tp.min(axis=0)[2]
                plane = np.stack([x, y, z])
                plane = np.reshape(plane, newshape=(3, -1)).T

                # make a hole at the intersection and behind
                model_center = np.mean(tp, axis=0)
                dis = np.linalg.norm(plane - model_center, axis=1)
                mask = dis > rg[1] / 2 * 0.75

                # tp = np.vstack((tp, plane[mask]))

                pc_.points = o3.utility.Vector3dVector(np.r_[tp, plane[mask]])
                pcs_processed.append(pc_)

                if self.flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = 1
                    draw_registration_result(source=pc, target=pcs_processed[-1], transformation=tf, window_name='Initial Setup')
        return pcs_processed

    def add_outliers(self, pcs, n_random):
        pcs_processed = []
        for pc in pcs:
            tp = np.asarray(pc.points)
            tp = copy.deepcopy(tp)
            """setup noise"""
            rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))  # range
            for n_random in self.num_random:
                pc_ = copy.deepcopy(pc)
                rands = (np.random.rand(n_random, 3) - 0.5) * rg + tp.mean(axis=0)
                pc_.points = o3.utility.Vector3dVector(np.r_[tp + rands])
                pcs_processed.append(pc_)
                if self.flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = 1
                    draw_registration_result(source=pc, target=pcs_processed[-1], transformation=tf, window_name='Initial Setup')
        assert len(pcs_processed) / pcs == len(self.voxel_sizes)
        return pcs_processed

    def down_sampling(self, pcs):
        pcs_processed = []
        for pc in pcs:
            tp = np.asarray(pc.points)
            tp = copy.deepcopy(tp)
            for voxel_size in self.voxel_sizes:
                pc_ = copy.deepcopy(pc)
                pc_down = pc_.voxel_down_sample(voxel_size)
                pcs_processed.append(pc_down)
                if self.flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = 1
                    draw_registration_result(source=pc, target=pcs_processed[-1], transformation=tf,
                                             window_name='Initial Setup')
        assert len(pcs_processed) / pcs == len(self.voxel_sizes)
        return pcs_processed

    def cutoff(self, pc):
        pass


def main():
    data_path = './data/TUW_TUW_models/TUW_models/'
    output_path = './data/TUW_TUW_data/'
    ds = dataset()
    ds.read(data_path)
    ds.generate(output_path)


if __name__ == '__main__':
    main()
