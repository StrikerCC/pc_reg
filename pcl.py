#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pcl.py
@Contact :   chengc0611@gmail.com
@License :   

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
7/22/2021 5:55 PM   Cheng CHen    1.0         None
'''

# import lib
import pcl
import numpy as np

p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
seg = p.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
indices, model = seg.segment()