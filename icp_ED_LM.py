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
    # Define point types
    PointType = itk.Point[itk.F, dimension]
    PointSetType = itk.PointSet[itk.F, dimension]

    # define transform type
    TransformType = itk.Euler3DTransform

    # define metric type
    PointSetMetricType = itk.EuclideanDistancePointSetToPointSetMetricv4[PointSetType]

    ShiftScalesType = itk.RegistrationParameterScalesFromPhysicalShift[PointSetMetricType]

    OptimizerType = itk.LevenbergMarquardtOptimizer

    RegistrationType = itk.PointSetToPointSetRegistrationMethod[PointSetType]

    return TransformType, PointSetMetricType, ShiftScalesType, OptimizerType, RegistrationType


def reg_object_init(TransformType,
                    PointSetMetricType, fixed_set, moving_set, transform,
                    ShiftScalesType,
                    OptimizerType, num_iterations,
                    RegistrationType):

    transform = TransformType.New()

    metric = PointSetMetricType.New()

    # shift_scale_estimator = ShiftScalesType.New(
    #     Metric=metric,
    #     VirtualDomainPointSet=metric.GetVirtualTransformedPointSet())
    scales = OptimizerType.ScaleType(transform.GetNumberOfParameters())
    optimizer = OptimizerType.New(
        UseCostFunctionGradient=False,
        Scales=scales,
        NumberOfIterations=num_iterations,
        ValueTolerance=0.0001,
        GradientTolerance=0.0001,
        EpsilonFunction=1e-5,
    )

    registration = RegistrationType.New(
        Metric=metric,
        Optimizer=optimizer,
        SetTransform=transform,
        SetFixedPointSet=fixed_set,
        SetMovingPointSet=moving_set)

    return registration
