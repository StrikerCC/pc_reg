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
        self.flag_show = False

        self.voxel_sizes = (0.7,)
        self.angles_cutoff_along = (0.0,)
        self.plane_sizes = (0.8,)
        self.Gaussian_sigma_factor = (0.2,)
        self.n_move = 2
        self.translation_rg_factor = (-2.1, 2.1)
        self.rotation_reg = (-0.1, 0.0)
        self.num_random = (100,)

        # self.voxel_sizes = (0.07, 0.67)
        # self.voxel_sizes = np.arange(self.voxel_sizes[0], self.voxel_sizes[1], (self.voxel_sizes[1]-self.voxel_sizes[0])/4)
        #
        # self.angles_cutoff_along = (0.0, 360.0)
        # self.angles_cutoff_along = np.arange(self.angles_cutoff_along[0], self.angles_cutoff_along[1], 60.0)
        # self.angle_cutoff = 90
        #
        # self.plane_sizes = (0.8, 1.7)
        # self.plane_sizes = np.arange(self.plane_sizes[0], self.plane_sizes[1], 0.8)
        #
        # self.Gaussian_sigma_factor = (0.2, 1.5)
        # self.Gaussian_sigma_factor = np.arange(self.Gaussian_sigma_factor[0], self.Gaussian_sigma_factor[1], 0.4)
        #
        # self.n_move = 16
        # self.translation_rg_factor = (-2.5, 2.5)
        # self.rotation_reg = (-360.0, 360.0)
        #
        # self.num_random = np.arange(50, 250, 100)

        self.instances = ['bunny', 'water_boiler', 'cisco_phone', 'red_mug_white_spots', 'strands_mounting_unit',
                          'burti', 'skull', 'yellow_toy_car', 'fruchtmolke', 'canon_camera_bag', 'dragon_recon',
                          'happy_recon'] #, 'lucy']
        # self.instances = ['bunny', 'water_boiler']

        self.instance_in_meter = {'bunny', 'water_boiler', 'cisco_phone', 'red_mug_white_spots',
                                  'strands_mounting_unit', 'burti', 'skull', 'yellow_toy_car', 'fruchtmolke',
                                  'canon_camera_bag', 'dragon_recon', 'happy_recon'}  # , 'lucy']

        self.instances_plane = {'bunny', 'water_boiler', 'cisco_phone', 'red_mug_white_spots',
                                'strands_mounting_unit', 'burti', 'skull', 'yellow_toy_car', 'fruchtmolke',
                                'canon_camera_bag', 'dragon_recon', 'happy_recon'}

        self.data_info = {'pc_model': None, 'pc_artificial': None, 'instance': None, 'unit': '', 'voxel_size': None,
                          'angle': None, 'pose': [], 'sigma': None, 'outliers': None, 'plane': None}

    # def read_instance(self, dir_path):
    #     self.dir_path_read = dir_path
    #     # instances = os.listdir(dir_path)
    #     instances = self.instances
    #     self.file_paths = [dir_path + instance + '/3D_model.pcd' for instance in instances]


class writer(dataset):
    def __init__(self):
        dataset.__init__(self)
        self.filename_len = 6
        self.meter_2_mm = True

    def write(self, sample_dir_path, output_dir_path, json_path):
        # setup output path and file
        if not os.path.isdir(output_dir_path):
            os.makedirs(output_dir_path)
        else:
            shutil.rmtree(output_dir_path)
            os.makedirs(output_dir_path)

        instances = self.instances
        self.file_paths = [sample_dir_path + instance + '/3D_model.pcd' for instance in instances]

        """why brother registering parameters to build artificial data layer after layer, 
        why not build the point cloud in place. Think about how many point cloud will be here after 5 different 
        downsampling, 6 different cutoff, 4 different plane added, 8 different noise level, it will blow up memory"""
        # register sample pcs and artificial info
        sources = self.__reg_pc(self.file_paths, self.instances)

        # peek model data
        print('Get model point cloud for')
        for i, source in enumerate(sources):
            print(i, source['instance'])

        # register artificial pc parameters
        print('Setting up artificial point cloud for')
        sources = self.__reg_down_sampling(sources)
        sources = self.__reg_cutoff(sources)
        # sources = self.__reg_add_outliers(sources)
        sources = self.__reg_add_plane(sources)
        sources = self.__reg_add_noise(sources)
        sources = self.__reg_add_pose(sources)
        print('Set up artificial point cloud for')

        # create artificial pc and save it
        print('Creating artificial point cloud')
        instance_cur = None
        for i, source in enumerate(sources):
            # read pc, down sampling, and so on
            self.__load_pc(source, flag_show=self.flag_show)

            if not instance_cur or not instance_cur == source['instance']:  # notice if instance change
                instance_cur = source['instance']
                print('     ', i, 'iter: Working on', source['instance'], source['pc_artificial'], ' range from',
                      source['pc_artificial'].get_min_bound(), 'to', source['pc_artificial'].get_max_bound())

            self.__exe_down_sampling(source, flag_show=self.flag_show)
            self.__exe_cutoff(source, flag_show=self.flag_show)
            # self.__exe_add_outliers(source, flag_show=False)
            self.__exe_add_plane(source, flag_show=self.flag_show)
            self.__exe_add_noise(source, flag_show=self.flag_show)
            self.__exe_add_pose(source, flag_show=self.flag_show)
            self.__save_pc(output_dir_path, index=i, source=source)
        print('Created artificial point cloud for\n')

        # make json file to retrieve data
        sources_record = sources  # self.__format_json(sources)
        with open(json_path, 'w') as f:
            json.dump(sources_record, f)

    def __reg_pc(self, file_paths, instances):
        sources = []
        for file_path, instance in zip(file_paths, instances):
            tf = np.eye(4)
            # transform the point cloud if necessary
            if 'bunny' in file_path:
                tf[:3, :3] = t3d.euler.euler2mat(*np.deg2rad([90.0, 0.0, 0.0]))
                # tf[:3, 3] = 0
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

            source = copy.deepcopy(self.data_info)
            source['pc_model'] = file_path
            source['instance'] = instance
            source['unit'] = 'mm'
            source['pose'].append(tf)
            sources.append(source)
            # print(file_path, "\n    max bounds for geometry coordinates", pc.get_max_bound())
        return sources

    def __load_pc(self, source, flag_show=False):
        file_path = source['pc_model']
        pc = o3.io.read_point_cloud(filename=file_path)
        if source['instance'] in self.instance_in_meter:
            pc.scale(scale=1000.0, center=pc.get_center())
        tf = source['pose'][0]
        pc.transform(tf)
        source['pc_artificial'] = pc

        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Read ' + str(source['instance']))

    def __save_pc(self, save_dir, index, source):
        filename_save = str(index) + '.pcd'
        filename_save = '0' * (self.filename_len - len(str(index))) + filename_save
        o3.io.write_point_cloud(save_dir + filename_save, source['pc_artificial'])
        del source['pc_artificial']  # get rid of the point cloud to save memory
        source['pc_artificial'] = save_dir + filename_save  # record the artificial pc saving address

    def __reg_down_sampling(self, sources):
        sources_processed = []
        for source in sources:
            for voxel_size in self.voxel_sizes:
                source_ = copy.deepcopy(source)
                source_['voxel_size'] = voxel_size
                sources_processed.append(source_)
        print('Down sampling', self.voxel_sizes)
        assert len(sources_processed) / len(sources) == len(self.voxel_sizes), str(len(sources_processed)) + ' ' + str(
            len(sources)) + ' ' + str(len(self.voxel_sizes))
        return sources_processed

    def __exe_down_sampling(self, source, flag_show=True):
        pc = source['pc_artificial']
        voxel_size = source['voxel_size']
        source['pc_artificial'] = pc.voxel_down_sample(voxel_size)
        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Initial Setup down to ' + str(voxel_size))

    def __reg_cutoff(self, sources, flag_show=True):
        sources_processed = []
        for source in sources:
            for angle_cutoff_along in self.angles_cutoff_along:
                source_ = copy.deepcopy(source)
                source_['angle'] = angle_cutoff_along
                sources_processed.append(source_)
        assert len(sources_processed) / len(sources) == len(self.angles_cutoff_along), str(
            len(sources_processed)) + ' ' + str(len(sources)) + ' ' + str(len(self.angles_cutoff_along))
        print('Cut off ', self.angles_cutoff_along)
        return sources_processed

    def __exe_cutoff(self, source, flag_show=True):
        angle_cutoff_along = np.deg2rad(source['angle'])
        pc = source['pc_artificial']
        tp = np.asarray(pc.points)
        model_center = np.mean(tp, axis=0)
        tp = tp - model_center
        normal = np.array([-np.sin(angle_cutoff_along), np.cos(angle_cutoff_along), 0.0]).T
        # plane_vector = np.array([np.cos(angle_cutoff_along), np.sin(angle_cutoff_along), 0.0]).T
        # plane_vector = normal
        mask = np.dot(tp, normal)
        mask = mask < 0
        tp = tp[mask, :]
        pc.points = o3.utility.Vector3dVector(tp)
        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Cutoff at ' + str(angle_cutoff_along))

    def __reg_add_pose(self, sources):
        sources_processed = []
        for source in sources:
            for i in range(self.n_move):
                source_ = copy.deepcopy(source)
                translation = ((self.translation_rg_factor[1] - self.translation_rg_factor[0]) * np.random.random(3) +
                               self.translation_rg_factor[0])
                orientation = (self.rotation_reg[1] - self.rotation_reg[0]) * np.random.random((3, 1)) + \
                              self.rotation_reg[0]
                tf = np.identity(4)
                tf[:3, :3] = t3d.euler.euler2mat(*orientation)
                tf[:3, 3] = translation
                # source_['pose'] = np.matmul(tf, source_['pose'])
                source_['pose'].append(tf)
                sources_processed.append(source_)
        print('# of Pose', self.n_move)
        return sources_processed

    def __exe_add_pose(self, source, flag_show=False):
        pc = source['pc_artificial']
        pc_init = copy.deepcopy(pc) if flag_show else None
        # change the pose according to model size
        tf = source['pose'][-1]
        tp = np.asarray(pc.points)
        rg = tp.max(axis=0) - tp.min(axis=0)  # range
        tf[:3, 3] = rg * tf[:3, 3]
        source['pose'][-1] = tf
        del tf
        pc.transform(source['pose'][-1])

        # reformat poses to make final poseqq
        tf_final = np.eye(4)
        for tf in source['pose']:
            tf_final = np.matmul(tf, tf_final)
        source['pose'] = tf_final.tolist()
        if flag_show:
            # pc_model = o3.io.read_point_cloud(source['pc_model'])
            o3.visualization.draw_geometries([pc_init, pc], window_name='Move at ' + str(tf_final))

    def __reg_add_outliers(self, sources):
        sources_processed = []
        for source in sources:
            for n_random in self.num_random:
                source_ = copy.deepcopy(source)
                source_['outliers'] = n_random
                sources_processed.append(source_)
        assert len(sources_processed) / len(sources) == len(self.num_random), str(len(sources_processed)) + ' ' + str(
            len(sources)) + ' ' + str(len(self.num_random))
        print('# of outliers', self.num_random)
        return sources_processed

    def __exe_add_outliers(self, source, flag_show=True):
        pc = source['pc_artificial']
        tp = np.asarray(pc.points)
        """setup outliers"""
        rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))  # range
        n_random = source['outliers']
        rands = (np.random.rand(n_random, 3) - 0.5) * rg + tp.mean(axis=0)
        pc.points = o3.utility.Vector3dVector(np.r_[tp, rands])
        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Initial Setup add ' + str(n_random) + ' outliers')

    def __reg_add_plane(self, sources):
        sources_processed = []
        for source in sources:
            for plane_size in self.plane_sizes:
                source_ = copy.deepcopy(source)
                source_['plane'] = plane_size
                sources_processed.append(source_)
        print('Adding planes', self.plane_sizes)
        return sources_processed

    def __exe_add_plane(self, source, flag_show=True):
        if source['instance'] not in self.instances_plane: return
        dis_nearest_neighbor = source['voxel_size']
        plane_size = source['plane']
        pc = source['pc_artificial']
        tp = np.asarray(pc.points)
        # dis_nearest_neighbor = min(np.linalg.norm(tp - tp[0, :], axis=1)[2:])
        rg = 1.5 * (tp.max(axis=0) - tp.min(axis=0))  # range

        # add a plane underneath the model
        # dis_nearest_neighbor = dis_nearest_neighbor / plane_size
        nx = int(plane_size * rg[0] / dis_nearest_neighbor)
        ny = int(plane_size * rg[1] / dis_nearest_neighbor)
        x = np.linspace(-plane_size * rg[0], rg[0] * plane_size, nx)
        y = np.linspace(-plane_size * rg[1], rg[1] * plane_size, ny)
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
        pc.points = o3.utility.Vector3dVector(np.r_[tp, plane[mask]])

        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Initial Setup add ' + str(plane_size) + ' plane')

    def __reg_add_noise(self, sources):
        sources_processed = []
        for i, source in enumerate(sources):
            for j, noise_sigma in enumerate(self.Gaussian_sigma_factor):
                source_ = copy.deepcopy(source)
                source_['sigma'] = noise_sigma
                sources_processed.append(source_)
        print('Adding Gaussian noise', self.Gaussian_sigma_factor)
        return sources_processed

    def __exe_add_noise(self, source, flag_show=True):
        noise_sigma = source['sigma']
        pc = source['pc_artificial']
        tp = np.asarray(pc.points)
        pc.points = o3.utility.Vector3dVector(np.r_[tp + noise_sigma * np.random.randn(*tp.shape)])
        if flag_show:
            o3.visualization.draw_geometries([pc], window_name='Initial Setup add ' + str(noise_sigma) + ' sigma')


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
        # process data
        if isinstance(source['pc_artificial'], str):
            source['pc_artificial'] = o3.io.read_point_cloud(source['pc_artificial'])  # read artificial pc
        if isinstance(source['pc_model'], str):
            pc_model = o3.io.read_point_cloud(source['pc_model'])
            pc_model.scale(scale=1000.0, center=pc_model.get_center())
            source['pc_model'] = pc_model
        if not isinstance(source['pose'], np.ndarray):
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
    for i in range(len(dl)):
        data = dl[i]
        pc_model = data['pc_model']
        pc_artificial = data['pc_artificial']
        tf = data['pose']
        pc_model_ = copy.deepcopy(pc_model)
        pc_model_.transform(tf)

        draw_registration_result(source=pc_artificial)
        draw_registration_result(source=pc_artificial, target=pc_model_)
        # print(dl[0])


if __name__ == '__main__':
    main()
