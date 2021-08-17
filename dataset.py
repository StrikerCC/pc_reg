# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/13/21 5:44 PM
"""

import copy
import os
# import glob
import shutil
import numpy as np
import json

import open3d as o3
import transforms3d as t3d
from vis import draw_registration_result


class dataset:
    def __init__(self):
        self.meter_2_mm = True
        self.flag_show = True
        self.voxel_sizes = (0.07, 0.67)
        self.voxel_sizes = np.arange(self.voxel_sizes[0], self.voxel_sizes[1], (self.voxel_sizes[1]-self.voxel_sizes[0])/5)

        self.angles_cutoff_along = (0.0, 360.0)
        self.angles_cutoff_along = np.arange(self.angles_cutoff_along[0], self.angles_cutoff_along[1], 120)
        self.angle_cutoff = 90

        self.plane_sizes = (1.5, 2.5)
        self.plane_sizes = np.arange(self.plane_sizes[0], self.plane_sizes[1], 0.5)

        self.Gaussian_sigma_factor = (0.2, 1.2)
        self.Gaussian_sigma_factor = np.arange(self.Gaussian_sigma_factor[0], self.Gaussian_sigma_factor[1], 0.5)

        self.n_move = 10
        self.translation_rg_factor = (0.8, 2.5)
        self.rotation_reg = (0.0, 180.0)

        self.num_random = np.arange(50, 250, 100)
        # self.dir_path_read = None
        self.file_paths = None
        self.instances = ['bunny', 'water_boiler', 'cisco_phone', 'red_mug_white_spots', 'strands_mounting_unit',
                          'burti', 'skull', 'yellow_toy_car', 'fruchtmolke', 'canon_camera_bag', 'dragon_recon',
                          'happy_recon', 'lucy']
        self.data_info = {'pc': None, 'instance': None, 'unit': None, 'voxel_size':  None, 'angle': None, 'pose': None,
                          'sigma': None, 'outliers': None, 'plane': None}

    # def read_instance(self, dir_path):
    #     self.dir_path_read = dir_path
    #     # instances = os.listdir(dir_path)
    #     instances = self.instances
    #     self.file_paths = [dir_path + instance + '/3D_model.pcd' for instance in instances]


class writer(dataset):
    def __init__(self):
        dataset.__init__(self)
        self.meter_2_mm = True

    def write(self, sample_dir_path, output_dir_path, json_path):
        # setup output path and file
        if not os.path.isdir(output_dir_path):
            os.makedirs(output_dir_path)
        else:
            shutil.rmtree(output_dir_path)
            os.makedirs(output_dir_path)

        # read sample pcs
        # instances = os.listdir(dir_path)
        instances = self.instances
        self.file_paths = [sample_dir_path + instance + '/3D_model.pcd' for instance in instances]

        # operate pcs
        sources = self.load_pc(self.file_paths, self.instances)[:1]
        sources = self.down_sampling(sources, flag_show=False)
        sources = self.cutoff(sources, flag_show=False)
        # sources = self.add_outliers(sources, flag_show=False)
        sources = self.add_plane(sources, flag_show=False)
        sources = self.add_pose(sources, flag_show=False)
        sources = self.add_noise(sources, flag_show=False)

        self.save_pc(output_dir_path, sources)
        with open(json_path, 'w') as f:
            json.dump(sources, f)

    def load_pc(self, file_paths, instances):
        sources = []
        for file_path, instance in zip(file_paths, instances):
            pc = o3.io.read_point_cloud(filename=file_path)
            if self.meter_2_mm: pc.scale(scale=1000.0, center=pc.get_center())
            tf = np.eye(4)
            # transform the point cloud if necessary
            if 'bunny' in file_path:
                tf[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([90.0, 0.0, 0.0]))
                tf[:3, 3] = 0
            elif 'skull' in file_path:
                tf[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([0.0, 90.0, 0.0]))
                tf[:3, 3] = 0
            elif 'burti' in file_path:
                tf[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([90.0, 0.0, 0.0]))
                tf[:3, 3] = 0
            elif 'dragon_recon' in file_path:
                tf[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([90.0, 0.0, 0.0]))
                tf[:3, 3] = 0
            elif 'happy_recon' in file_path:
                tf[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([90.0, 0.0, 0.0]))
                tf[:3, 3] = 0
            pc.transform(tf)
            source = copy.deepcopy(self.data_info)
            source['pc'] = pc
            source['instance'] = instance
            source['unit'] = 'mm'
            sources.append(source)
            print(file_path, "\n    max bounds for geometry coordinates", pc.get_max_bound())
        return sources

    def save_pc(self, save_dir, sources):
        for i, source in enumerate(sources):
            filename_len = 6
            filename = str(i) + '.pcd'
            filename = '0' * (filename_len - len(str(i))) + filename
            pc = source['pc']
            o3.io.write_point_cloud(save_dir + filename, pc)
            source['pc'] = save_dir + filename

    def down_sampling(self, sources, flag_show=True):
        sources_processed = []
        for source in sources:
            pc = source['pc']
            # tp = np.asarray(pc.points)
            # tp = copy.deepcopy(tp)
            for voxel_size in self.voxel_sizes:
                source_ = copy.deepcopy(source)
                pc_ = source_['pc']
                pc_ = pc_.voxel_down_sample(voxel_size)
                source_['voxel_size'] = voxel_size
                sources_processed.append(source_)
                if flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = pc_.get_max_bound()
                    draw_registration_result(source=pc, target=pc_, transformation=tf,
                                             window_name='Initial Setup down to ' + str(voxel_size))
        assert len(sources_processed) / len(sources) == len(self.voxel_sizes), str(len(sources_processed)) + ' ' + str(len(sources)) + ' ' + str(len(self.voxel_sizes))
        return sources_processed

    def cutoff(self, sources, flag_show=True):
        sources_processed = []
        for source in sources:
            pc = source['pc']
            tp = np.asarray(pc.points)
            tp = copy.deepcopy(tp)
            model_center = np.mean(tp, axis=0)
            tp = tp - model_center

            for angle_cutoff_along in self.angles_cutoff_along:
                source_ = copy.deepcopy(source)
                pc_ = source_['pc']
                tp_ = np.asarray(pc_.points)

                # normal = np.array([np.cos(angle_cutoff_along), np.sin(angle_cutoff_along)])
                plane_vector = np.array([-np.sin(angle_cutoff_along), np.cos(angle_cutoff_along), 0]).T
                mask = np.dot(tp, plane_vector)
                mask = mask < 0
                tp_ = tp_[mask, :]
                pc_.points = o3.utility.Vector3dVector(tp_)

                source_['angle'] = angle_cutoff_along
                sources_processed.append(source_)
                if flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = pc_.get_max_bound()
                    draw_registration_result(source=pc, target=pc_, transformation=tf,
                                             window_name='Cutoff at ' + str(angle_cutoff_along))
        assert len(sources_processed) / len(sources) == len(self.angles_cutoff_along), str(len(sources_processed)) + ' ' + str(len(sources)) + ' ' + str(len(self.angles_cutoff_along))
        return sources_processed

    def add_pose(self, sources, flag_show=False):
        sources_processed = []
        for source in sources:
            pc = source['pc']
            tp = np.asarray(pc.points)
            tp = copy.deepcopy(tp)
            for i in range(self.n_move):
                source_ = copy.deepcopy(source)
                pc_ = source_['pc']
                translation = self.translation_rg_factor[0] + np.random.random((3)) * self.translation_rg_factor[1]
                orientation = np.random.random((3, 1)) * self.rotation_reg[1]
                ans = np.identity(4)
                ans[:3, :3] = t3d.euler.euler2mat(*orientation)
                ans[:3, 3] = translation
                pc_.transform(ans)
                source_['pose'] = ans.tolist()
                sources_processed.append(source_)
                if flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = pc_.get_max_bound()
                    draw_registration_result(source=pc, target=pc_, transformation=tf,
                                             window_name='Move at ' + str(ans))
        return sources_processed

    def add_outliers(self, sources, flag_show=True):
        sources_processed = []
        for source in sources:
            pc = source['pc']
            tp = np.asarray(pc.points)
            tp = copy.deepcopy(tp)
            """setup noise"""
            rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))  # range
            for n_random in self.num_random:
                source_ = copy.deepcopy(source)
                pc_ = source_['pc']
                rands = (np.random.rand(n_random, 3) - 0.5) * rg + tp.mean(axis=0)
                pc_.points = o3.utility.Vector3dVector(np.r_[tp, rands])
                source_['outliers'] = n_random
                sources_processed.append(source_)
                if flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = pc_.get_max_bound()
                    draw_registration_result(source=pc, target=pc_, transformation=tf,
                                         window_name='Initial Setup add ' + str(n_random) + ' outliers')
        assert len(sources_processed) / len(sources) == len(self.num_random), str(len(sources_processed)) + ' ' + str(len(sources)) + ' ' + str(len(self.num_random))
        return sources_processed

    def add_plane(self, sources, flag_show=True):
        sources_processed = []
        for source in sources:
            pc = source['pc']
            tp = np.asarray(pc.points)
            tp = copy.deepcopy(tp)
            dis_nearest_neighbor = min(np.linalg.norm(tp - tp[0, :], axis=1)[2:])
            rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))  # range

            for plane_size in self.plane_sizes:
                source_ = copy.deepcopy(source)
                pc_ = source_['pc']

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

                source_['plane'] = plane_size
                sources_processed.append(source_)

                if flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = pc_.get_max_bound()
                    draw_registration_result(source=pc, target=pc_, transformation=tf,
                                             window_name='Initial Setup add ' + str(plane_size) + ' plane')
        return sources_processed

    def add_noise(self, sources, flag_show=True):
        sources_processed = []
        for source in sources:
            pc = source['pc']
            tp = np.asarray(pc.points)

            """setup noise"""
            for noise_sigma in self.Gaussian_sigma_factor:
                source_ = copy.deepcopy(source)
                pc_ = source_['pc']
                pc_.points = o3.utility.Vector3dVector(np.r_[tp + noise_sigma * np.random.randn(*tp.shape)])
                source_['sigma'] = noise_sigma
                sources_processed.append(source_)
                if flag_show:
                    tf = np.eye(4)
                    tf[:3, 3] = pc_.get_max_bound()
                    draw_registration_result(source=pc, target=pc_, transformation=tf,
                                             window_name='Initial Setup add ' + str(noise_sigma) + ' sigma')
        return sources_processed


class reader(dataset):
    def __init__(self):
        dataset.__init__(self)
        self.meter_2_mm = False
        self.sources = None

    def read(self, json_path):
        with open(json_path, 'r') as f:
            self.sources = json.load(f)

        # check unit
        if self.sources[0]['unit'] == 'mm':
            self.meter_2_mm = False
        elif self.sources[0]['unit'] == 'm':
            self.meter_2_mm = True
        else:
            raise AssertionError('Unknown unit ', self.sources[0]['unit'])

    def __len__(self):
        return len(self.sources) if self.sources else 0

    def __getitem__(self, item):
        if not self.sources: return None
        source = self.sources[item]
        source['pc'] = o3.io.read_point_cloud(source['pc'])
        source['pose'] = np.asarray(source['pose'])
        return source


def main():
    sample_path = './data/TUW_TUW_models/TUW_models/'
    output_path = './data/TUW_TUW_data/'
    output_json_path = output_path + 'data.json'
    ds = writer()
    # ds.read_instance(data_path)
    ds.write(sample_path, output_path, output_json_path)

    dl = reader()
    dl.read(output_json_path)
    print(dl[0])


if __name__ == '__main__':
    main()
