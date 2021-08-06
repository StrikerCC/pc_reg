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


def main():
    a = np.random.random((10000, 2))
    noise = np.random.normal(loc=[0.0, 10.0], scale=[1.0, 4.0], size=a.shape)
    print(a.shape)
    print(noise.shape)
    print(np.mean(noise, axis=0))
    print(np.std(noise, axis=0))


if __name__ == '__main__':
    main()