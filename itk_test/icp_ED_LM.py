#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   icp_ED_LM.py
@Contact :   chengc0611@gmail.com
@License :   

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
7/29/2021 3:13 PM   Cheng Chen    1.0         None
'''

# import lib
import itk


def reg_object_type_def(dimension=3):
    itk_ = itk
    # objs = itk_test
    # Define point types
    PointType = itk.Point[itk.F, dimension]
    PointSetType = itk.PointSet[itk.F, dimension]

    # define transform type
    TransformType = itk.Euler3DTransform

    # define metric type
    PointSetMetricType = itk.EuclideanDistancePointSetToPointSetMetricv4[PointSetType]

    OptimizerType = itk.LevenbergMarquardtOptimizer

    RegistrationType = itk.PointSetToPointSetRegistrationMethod[PointSetType]
    # RegistrationType = itk_test.ITKRegistrationMethodsv4.ImageRegistrationMethodv4 #[PointSetType]

    return TransformType, PointSetMetricType, OptimizerType, RegistrationType


def reg_object_init(TransformType,
                    PointSetMetricType, fixed_set, moving_set,
                    OptimizerType, num_iterations,
                    RegistrationType):

    transform = TransformType.New()
    transform.SetIdentity()

    metric = PointSetMetricType.New()

    # scales = OptimizerType.ScaleType(transform.GetNumberOfParameters())
    optimizer = OptimizerType.New(
        UseCostFunctionGradient=False,
        # Scales=scales,
        NumberOfIterations=num_iterations,
        ValueTolerance=1e-5,
        GradientTolerance=1e-5,
        EpsilonFunction=1e-5,
    )

    # registration = RegistrationType.New(
    #     # InitialTransformParameters=transform.GetParameters(),
    #     Metric=metric,
    #     Optimizer=optimizer,
    #     SetTransform=transform,
    #     SetFixedPointSet=fixed_set,
    #     SetMovingPointSet=moving_set
    # )

    return transform, optimizer
